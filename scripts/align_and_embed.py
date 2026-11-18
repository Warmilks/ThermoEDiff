import argparse
from pathlib import Path
import math
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import torch
from transformers import AutoTokenizer, EsmModel

import parasail


VEC_DIM = 1280
VEC_DTYPE = np.float16

AA_OK = set("ACDEFGHIKLMNPQRSTVWYBXZJUO")

def sanitize(seq: str) -> str:
    seq = seq.strip().upper()
    return "".join(ch if ch in AA_OK else "X" for ch in seq)

def nw_align(a: str, b: str, gap_open: int = 10, gap_extend: int = 1):

    res = parasail.nw_trace_scan_16(a, b, gap_open, gap_extend, parasail.blosum62)
    qa = res.traceback.query
    qb = res.traceback.ref
    return qa, qb

@torch.no_grad()
def meanpool_residue_repr(model: torch.nn.Module, tokenizer, seqs, max_len: int, device: str):
    tok = tokenizer(
        seqs,
        return_tensors="pt",
        max_length=max_len,
        padding=True,
        truncation=True,
    )
    tok = {k: v.to(device) for k, v in tok.items()}
    out = model(**tok).last_hidden_state

    attn = tok["attention_mask"].bool()
    attn[:, 0] = False

    masked = out * attn.unsqueeze(-1)
    denom = attn.sum(dim=1).clamp(min=1).unsqueeze(-1)
    return masked.sum(dim=1) / denom

def open_writer(out_path: Path):
    schema = pa.schema([
        ("idx", pa.int64()),
        ("wt_seq", pa.string()),
        ("mut_seq", pa.string()),
        ("dir_vec", pa.list_(pa.float16(), list_size=VEC_DIM)),
    ])
    writer = pq.ParquetWriter(
        out_path.as_posix(),
        schema=schema,
        compression="zstd",
        use_dictionary=True,
    )
    return writer, schema

def flush(writer, schema, buf_idx, buf_wt, buf_mut, buf_vecs):
    if not buf_idx:
        return
    mat = np.stack(buf_vecs, axis=0).astype(VEC_DTYPE)
    values = pa.array(mat.reshape(-1), type=pa.float16())
    vec_col = pa.FixedSizeListArray.from_arrays(values, VEC_DIM)
    table = pa.Table.from_arrays(
        [
            pa.array(buf_idx, type=pa.int64()),
            pa.array(buf_wt, type=pa.string()),
            pa.array(buf_mut, type=pa.string()),
            vec_col,
        ],
        schema=schema,
    )
    writer.write_table(table)
    buf_idx.clear(); buf_wt.clear(); buf_mut.clear(); buf_vecs.clear()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_parquet", type=str, default="pairs_filtered.parquet")
    ap.add_argument("--out_dir", type=str, default="l2t_train")
    ap.add_argument("--model_name", type=str, default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--read_batch", type=int, default=4096)
    ap.add_argument("--embed_batch", type=int, default=64)
    ap.add_argument("--flush_n", type=int, default=4096)
    ap.add_argument("--shard_size", type=int, default=200000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = EsmModel.from_pretrained(args.model_name).to(device)
    model.eval()

    pf = pq.ParquetFile(args.in_parquet)

    shard_id = 0
    shard_path = out_dir / f"train_{shard_id:04d}.parquet"
    writer, schema = open_writer(shard_path)

    buf_idx, buf_wt, buf_mut, buf_vecs = [], [], [], []
    global_idx = 0
    shard_count = 0

    for batch in pf.iter_batches(batch_size=args.read_batch, columns=["meso_seg", "thermo_seg"]):
        meso_list = batch.column("meso_seg").to_pylist()
        thermo_list = batch.column("thermo_seg").to_pylist()


        aligned_wt = []
        aligned_mut = []
        ungapped_wt = []
        ungapped_mut = []

        for a, b in zip(meso_list, thermo_list):
            a = sanitize(a)
            b = sanitize(b)

            qa, qb = nw_align(a, b)
            aligned_wt.append(qa)
            aligned_mut.append(qb)

            ungapped_wt.append(qa.replace("-", ""))
            ungapped_mut.append(qb.replace("-", ""))


        for i0 in range(0, len(ungapped_wt), args.embed_batch):
            i1 = min(i0 + args.embed_batch, len(ungapped_wt))
            wt_chunk = ungapped_wt[i0:i1]
            mut_chunk = ungapped_mut[i0:i1]

            wt_emb = meanpool_residue_repr(model, tokenizer, wt_chunk, args.max_len, device)
            mut_emb = meanpool_residue_repr(model, tokenizer, mut_chunk, args.max_len, device)

            vec = (mut_emb - wt_emb).detach().cpu().numpy().astype(VEC_DTYPE)

            for j in range(i1 - i0):
                buf_idx.append(global_idx)
                buf_wt.append(aligned_wt[i0 + j])
                buf_mut.append(aligned_mut[i0 + j])
                buf_vecs.append(vec[j])
                global_idx += 1
                shard_count += 1

                if len(buf_idx) >= args.flush_n:
                    flush(writer, schema, buf_idx, buf_wt, buf_mut, buf_vecs)

                if shard_count >= args.shard_size:
                    flush(writer, schema, buf_idx, buf_wt, buf_mut, buf_vecs)
                    writer.close()

                    shard_id += 1
                    shard_path = out_dir / f"train_{shard_id:04d}.parquet"
                    writer, schema = open_writer(shard_path)
                    shard_count = 0

    flush(writer, schema, buf_idx, buf_wt, buf_mut, buf_vecs)
    writer.close()
    print(f"[OK] 完成。总样本数={global_idx}  输出目录={out_dir}")

if __name__ == "__main__":
    main()
