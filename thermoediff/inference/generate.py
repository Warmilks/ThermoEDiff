from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Set

import numpy as np
import torch
from transformers import AutoTokenizer

from thermoediff.models.diffusion import AlignDiff


AA20 = list("ACDEFGHIKLMNPQRSTVWY")


def _dict_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:

    from collections.abc import Mapping

    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, Mapping):
            moved = {}
            for kk, vv in v.items():
                moved[kk] = vv.to(device) if torch.is_tensor(vv) else vv
            out[k] = moved
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


@torch.no_grad()
def _meanpool_residue_repr(esm_model: torch.nn.Module, tokenizer, seqs: List[str], max_len: int) -> torch.Tensor:


    tok = tokenizer(
        seqs,
        return_tensors="pt",
        max_length=max_len,
        padding="longest",
        truncation=True,
        return_token_type_ids=False,
    )
    dev = next(esm_model.parameters()).device
    tok = {k: v.to(dev) for k, v in tok.items()}

    hs = esm_model(**tok).last_hidden_state

    attn = tok["attention_mask"].bool()
    attn[:, 0] = False

    lengths = tok["attention_mask"].sum(dim=1)
    eos_idx = (lengths - 1).clamp(min=0)
    batch_idx = torch.arange(attn.shape[0], device=dev)
    attn[batch_idx, eos_idx] = False

    denom = attn.sum(dim=1).clamp(min=1).unsqueeze(-1)
    pooled = (hs * attn.unsqueeze(-1)).sum(dim=1) / denom
    return pooled


@torch.no_grad()
def compute_y_ab(backbone: AlignDiff, tokenizer, seq_a: str, seq_b: str, max_len: int) -> torch.Tensor:


    esm = backbone.embedder.esm
    rep = _meanpool_residue_repr(esm, tokenizer, [seq_a, seq_b], max_len=max_len)
    y = rep[1] - rep[0]
    return y.float().detach().cpu()


def load_checkpoint(backbone: AlignDiff, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt


    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    backbone.load_state_dict(state, strict=True)
    backbone.to(device)
    backbone.eval()


def get_aa20_ids(tokenizer) -> List[int]:


    aa_ids = []
    for a in AA20:
        tid = tokenizer.convert_tokens_to_ids(a)
        if tid is None or tid == tokenizer.unk_token_id:
            raise RuntimeError(f"Tokenizer does not contain AA token '{a}'")
        aa_ids.append(int(tid))
    return aa_ids


def residue_positions(attn_mask_1d: torch.Tensor) -> torch.Tensor:


    idx = attn_mask_1d.nonzero(as_tuple=False).squeeze(-1)
    if idx.numel() < 2:
        return torch.empty(0, dtype=torch.long)
    return idx[1:-1]


def ids_to_seq_aa20(tokenizer, ids_1d: torch.Tensor, attn_mask_1d: torch.Tensor) -> str:


    pos = residue_positions(attn_mask_1d)
    if pos.numel() == 0:
        return ""
    toks = tokenizer.convert_ids_to_tokens(ids_1d[pos].tolist())

    out = []
    for t in toks:
        if len(t) == 1 and t in AA20:
            out.append(t)
        else:

            return ""
    return "".join(out)


def sample_from_logits_aa20(
    logits_1d: torch.Tensor,
    aa20_ids: List[int],
    temperature: float,
    top_k: int,
    top_p: float,
    generator: Optional[torch.Generator] = None,
) -> int:


    aa_logits = logits_1d[aa20_ids]
    t = max(float(temperature), 1e-6)
    aa_logits = aa_logits / t

    probs = torch.softmax(aa_logits, dim=-1)


    if top_k > 0 and top_k < probs.numel():
        vals, idxs = torch.topk(probs, k=top_k, dim=-1)
        probs2 = torch.zeros_like(probs)
        probs2[idxs] = vals
        probs = probs2 / probs2.sum()


    if 0.0 < top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cdf = torch.cumsum(sorted_probs, dim=-1)
        keep = cdf <= top_p
        keep[0] = True
        kept_idx = sorted_idx[keep]
        probs2 = torch.zeros_like(probs)
        probs2[kept_idx] = probs[kept_idx]
        probs = probs2 / probs2.sum()


    sampled = torch.multinomial(probs, num_samples=1, generator=generator).item()
    return int(aa20_ids[sampled])


def decode_candidate_edit_mode(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    tokenizer,
    aa20_ids: List[int],
    mut_k: int,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: Optional[torch.Generator] = None,
) -> str:


    pos = residue_positions(attn_mask)
    if pos.numel() == 0:
        return ""


    cand_ids = input_ids.clone()


    orig_ids = input_ids[pos]


    gains = []
    for p in pos.tolist():
        orig_id = int(input_ids[p].item())
        if orig_id not in aa20_ids:
            return ""
        aa_logits = logits[p, aa20_ids]
        orig_idx = aa20_ids.index(orig_id)
        orig_logit = aa_logits[orig_idx].item()


        best_alt = torch.max(torch.cat([aa_logits[:orig_idx], aa_logits[orig_idx+1:]]))
        gain = float(best_alt.item() - orig_logit)
        gains.append(gain)

    gains_t = torch.tensor(gains, dtype=torch.float32)

    K = int(mut_k)
    if not 0 <= K <= pos.numel():
        raise ValueError("Mutation count must be between zero and sequence length")
    if K == 0:

        return ids_to_seq_aa20(tokenizer, cand_ids, attn_mask)

    top_vals, top_idx = torch.topk(gains_t, k=K, largest=True)
    mut_positions = pos[top_idx.to(pos.device)]


    for p in mut_positions.tolist():
        orig_id = int(input_ids[p].item())

        alternative_ids = [token_id for token_id in aa20_ids if token_id != orig_id]
        cand_ids[p] = sample_from_logits_aa20(
            logits_1d=logits[p], aa20_ids=alternative_ids,
            temperature=temperature, top_k=top_k, top_p=top_p,
            generator=generator,
        )

    return ids_to_seq_aa20(tokenizer, cand_ids, attn_mask)


def decode_candidate_sample_mode(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
    tokenizer,
    aa20_ids: List[int],
    temperature: float,
    top_k: int,
    top_p: float,
    generator: Optional[torch.Generator] = None,
) -> str:


    pos = residue_positions(attn_mask)
    if pos.numel() == 0:
        return ""

    cand_ids = input_ids.clone()

    for p in pos.tolist():
        new_id = sample_from_logits_aa20(
            logits_1d=logits[p],
            aa20_ids=aa20_ids,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        cand_ids[p] = new_id

    return ids_to_seq_aa20(tokenizer, cand_ids, attn_mask)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--bert_dir", type=str, default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--out_root", type=str, default="generated_sequences")
    ap.add_argument("--steps", type=int, default=128)

    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)


    ap.add_argument("--mode", type=str, default="edit", choices=["edit", "sample"])
    ap.add_argument("--mut_k", type=int, default=1, help="number of substitutions in edit mode")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0, help="AA20-topk sampling (0 disables)")
    ap.add_argument("--top_p", type=float, default=1.0, help="AA20 nucleus sampling (1 disables)")
    ap.add_argument("--max_rounds", type=int, default=5, help="max attempts to fill unique candidates per C")


    ap.add_argument("--inSize", type=int, default=1280)
    ap.add_argument("--tknNum", type=int, default=33)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--A", type=int, default=2)
    ap.add_argument("--fcSize", type=int, default=1024)
    ap.add_argument("--std", type=float, default=0.27595767)

    args = ap.parse_args()
    if min(args.steps, args.num, args.batch, args.max_rounds) < 1:
        ap.error("steps, num, batch and max_rounds must be positive")
    if args.temperature <= 0 or args.mut_k < 1:
        ap.error("temperature and mut_k must be positive")

    plan_path = Path(args.plan)
    out_root = Path(args.out_root)
    ckpt_path = Path(args.ckpt)
    out_root.mkdir(parents=True, exist_ok=True)

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, trust_remote_code=True, do_lower_case=False)
    aa20_ids = get_aa20_ids(tokenizer)

    backbone = AlignDiff(
        bertDir=args.bert_dir,
        inSize=args.inSize,
        tknNum=args.tknNum,
        L=args.L,
        H=args.H,
        A=args.A,
        fcSize=args.fcSize,
        std=args.std,
    )
    load_checkpoint(backbone, ckpt_path, device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    parents = list(plan.keys())
    print(f"[INFO] Loaded plan: {len(parents)} parents from {plan_path}")
    print(f"[INFO] Out root: {out_root.resolve()}")
    print(f"[INFO] Mode={args.mode} | mut_k={args.mut_k} | T={args.temperature} | top_k={args.top_k} | top_p={args.top_p}")

    for p_i, parent in enumerate(parents, 1):
        item = plan[parent]
        A_seq = item["A"]["seq"]
        B_seq = item["B"]["seq"]
        D_seq = item.get("D", item["A"])["seq"]
        c_list = item["C_list"]

        parent_dir = out_root / parent
        parent_dir.mkdir(parents=True, exist_ok=True)

        y_ab = compute_y_ab(backbone, tokenizer, A_seq, B_seq, max_len=args.max_len)
        print(f"[{p_i}/{len(parents)}] Parent={parent} | C_count={len(c_list)}")

        for c_j, c_item in enumerate(c_list, 1):
            c_id = c_item["id"]
            C_seq = c_item["seq"].strip()
            if not C_seq or any(aa not in AA20 for aa in C_seq):
                raise ValueError(f"{c_id}: target must contain standard amino acids only")
            if len(C_seq) + 2 > args.max_len:
                raise ValueError(f"{c_id}: target exceeds max_len; increase max_len")
            out_file = parent_dir / f"{c_id}.txt"

            y_used = y_ab.unsqueeze(0)


            generated: List[str] = []
            seen: Set[str] = set()
            rounds = 0


            while len(generated) < args.num and rounds < args.max_rounds:
                rounds += 1

                cur = min(args.batch, args.num - len(generated))

                base_seed = args.seed + p_i * 100000 + c_j * 1000 + rounds
                torch.manual_seed(base_seed)
                np.random.seed(base_seed)


                g = torch.Generator(device=device)
                g.manual_seed(base_seed + 999)

                seqA_list = [C_seq] * cur
                seqB_list = [D_seq] * cur

                tokA = tokenizer(
                    seqA_list,
                    return_tensors="pt",
                    max_length=args.max_len,
                    padding="longest",
                    truncation=True,
                    return_token_type_ids=False,
                )
                tokB = tokenizer(
                    seqB_list,
                    return_tensors="pt",
                    max_length=args.max_len,
                    padding="longest",
                    truncation=True,
                    return_token_type_ids=False,
                )

                y_batch = y_used.repeat(cur, 1)

                data = {"tokenizedSeqAArr": tokA, "tokenizedSeqBArr": tokB, "tokenizedLabArr": y_batch}
                data = _dict_to_device(data, device)


                res = backbone(data, T=args.steps)
                logits_all = res["tkn_pre"]

                for k in range(cur):
                    logits = logits_all[k]
                    input_ids = data["tokenizedSeqAArr"]["input_ids"][k]
                    attn_mask = data["tokenizedSeqAArr"]["attention_mask"][k]

                    if args.mode == "edit":
                        seq = decode_candidate_edit_mode(
                            logits=logits,
                            input_ids=input_ids,
                            attn_mask=attn_mask,
                            tokenizer=tokenizer,
                            aa20_ids=aa20_ids,
                            mut_k=args.mut_k,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            generator=g,
                        )
                    else:
                        seq = decode_candidate_sample_mode(
                            logits=logits,
                            input_ids=input_ids,
                            attn_mask=attn_mask,
                            tokenizer=tokenizer,
                            aa20_ids=aa20_ids,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            top_p=args.top_p,
                            generator=g,
                        )


                    if not seq:
                        continue
                    if seq == C_seq:
                        continue
                    if seq in seen:
                        continue

                    seen.add(seq)
                    generated.append(seq)

                    if len(generated) >= args.num:
                        break

            if len(generated) < args.num:
                print(f"[WARN] {c_id}: generated {len(generated)}/{args.num} unique candidates")

            with open(out_file, "w", encoding="utf-8") as fw:
                fw.write(f"# parent: {parent}\n")
                fw.write(f"# A_id: {item['A']['id']}\n# B_id: {item['B']['id']}\n# D_id: {item.get('D', item['A'])['id']}\n")
                fw.write(f"# A_seq: {A_seq}\n# B_seq: {B_seq}\n# D_seq: {D_seq}\n")
                fw.write(f"# C_id: {c_id}\n# C_seq: {C_seq}\n")
                fw.write("# y: meanpool(B)-meanpool(A)\n")
                fw.write(f"# steps: {args.steps}\n")
                fw.write(f"# mode: {args.mode}\n")
                if args.mode == "edit":
                    fw.write(f"# mut_k: {args.mut_k}\n")
                fw.write(f"# temperature: {args.temperature} | top_k: {args.top_k} | top_p: {args.top_p}\n")
                fw.write(f"# unique_generated: {len(generated)} (rounds={rounds})\n")
                for s in generated[:args.num]:
                    fw.write(s + "\n")

            if c_j % 10 == 0 or c_j == len(c_list):
                print(f"  - C[{c_j}/{len(c_list)}] wrote {out_file.name} | unique={len(generated)}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
