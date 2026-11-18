import argparse
from pathlib import Path
import duckdb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddb", type=str, default="learn2therm.ddb", help="learn2therm.ddb 路径")
    ap.add_argument("--out", type=str, default="pairs_filtered.parquet", help="输出 parquet 路径")
    ap.add_argument("--target_n", type=int, default=4_000_000)
    ap.add_argument("--cov", type=float, default=0.95)
    ap.add_argument("--thermo_min", type=float, default=60.0)
    ap.add_argument("--meso_max", type=float, default=40.0)
    ap.add_argument("--deltaT", type=float, default=20.0)
    ap.add_argument("--len_diff_frac", type=float, default=0.10)
    ap.add_argument("--evalue_max", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(args.ddb, read_only=True)
    con.execute(f"PRAGMA threads={args.threads}")


    base_cte = f"""
    WITH filtered AS (
        SELECT
            p.meso_pid,
            p.thermo_pid,
            p.meso_taxid,
            p.thermo_taxid,
            p.query_align_start, p.query_align_end, p.query_align_len, p.query_align_cov,
            p.subject_align_start, p.subject_align_end, p.subject_align_len, p.subject_align_cov,
            p.local_E_value,
            p.bit_score,
            tm.temperature AS meso_temp,
            tt.temperature AS thermo_temp,
            substring(pm.protein_seq, p.query_align_start, p.query_align_len) AS meso_seg,
            substring(pt.protein_seq, p.subject_align_start, p.subject_align_len) AS thermo_seg,
            length(pm.protein_seq) AS meso_len,
            length(pt.protein_seq) AS thermo_len
        FROM pairs p
        JOIN proteins pm ON pm.pid = p.meso_pid
        JOIN proteins pt ON pt.pid = p.thermo_pid
        JOIN taxa tm ON tm.taxid = p.meso_taxid
        JOIN taxa tt ON tt.taxid = p.thermo_taxid
        WHERE p.query_align_cov >= {args.cov}
          AND p.subject_align_cov >= {args.cov}
          AND p.local_E_value <= {args.evalue_max}
          AND tt.temperature >= {args.thermo_min}
          AND tm.temperature <= {args.meso_max}
          AND (tt.temperature - tm.temperature) >= {args.deltaT}
          AND abs(length(pm.protein_seq) - length(pt.protein_seq))
                <= {args.len_diff_frac} * greatest(length(pm.protein_seq), length(pt.protein_seq))
    )
    """

    select_try_sample = base_cte + f"""
    SELECT * FROM filtered
    USING SAMPLE {args.target_n} ROWS
    """

    select_fallback = base_cte + f"""
    SELECT * FROM filtered
    ORDER BY random()
    LIMIT {args.target_n}
    """

    sql_try_sample = f"""
    COPY (
    {select_try_sample}
    ) TO '{out_path.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    sql_fallback = f"""
    COPY (
    {select_fallback}
    ) TO '{out_path.as_posix()}'
    (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    try:
        con.execute(sql_try_sample)
    except Exception as e:
        print("[WARN] USING SAMPLE 失败，改用 ORDER BY random() LIMIT ...")
        print("原因：", repr(e))
        con.execute(sql_fallback)

    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_path.as_posix()}')").fetchone()[0]
    print(f"[OK] 导出完成：{out_path}  行数={n}")

if __name__ == "__main__":
    main()
