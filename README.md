# ThermoEDiff

Implementation of **ThermoEDiff: An Evolution-Direction-Conditioned Diffusion Method for Protein Thermostability Optimization**.

ThermoEDiff conditions residual diffusion on ESM2 protein representations and an evolutionary direction computed from a reference sequence pair. Fixed-budget decoding selects mutation positions and samples amino acid substitutions to generate candidate sequences.

## Files

- `model.py` — ESM2 encoder, conditional diffusion network and amino acid head.
- `training.py` and `train.py` — joint denoising/reconstruction loss and single-process or distributed training.
- `stream_dataset.py` and `collate.py` — streaming Parquet input and tokenization.
- `generate.py` — reference-direction computation, diffusion sampling and candidate decoding.
- `metrics.py` — training metric utilities.

## Installation

Use Python 3.10 or newer. Install a PyTorch build appropriate for your CUDA environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

The encoder is `facebook/esm2_t33_650M_UR50D` from Hugging Face. GPU execution is recommended. This repository contains source code only; training data and trained ThermoEDiff checkpoints are not included.

## Training

Prepare Parquet shards with columns `wt_seq`, `mut_seq` and `dir_vec`. Each row contains a position-aligned reference sequence pair and its 1280-dimensional direction vector. Sequence pairs must produce equal token lengths. Configure the shard pattern and hyperparameters in `train.py`.

```bash
python train.py
# Multi-GPU training on Linux with CUDA/NCCL:
torchrun --standalone --nproc_per_node=4 train.py
```

Defaults include a batch size of 64 per GPU, eight approximate epochs, learning rate 0.0004 and warmup ratio 0.02. Distributed streaming partitions files across ranks and workers; provide enough populated shards for all workers. Saved checkpoints use the `ThermoEDiff` filename prefix.

## Candidate generation

Create a JSON input plan. `A` and `B` are the reference pair ordered from lower to higher thermostability, and `C_list` contains target sequences. The sequences below illustrate the input format only.

```json
{
  "example": {
    "A": {"id": "reference_A", "seq": "ACDEFGHIKLMNPQRSTVWY"},
    "B": {"id": "reference_B", "seq": "ACDEYGHIKLMNPQRSTVWY"},
    "C_list": [{"id": "target_1", "seq": "ACDEFGHIKLMNPQRSTVWY"}]
  }
}
```

Run with your trained checkpoint:

```bash
python generate.py --plan plan.json --ckpt /path/to/checkpoint.pkl --mode edit --mut_k 3 --num 10 --steps 128 --max_rounds 5 --temperature 1.0 --out_root generated_sequences
```

Each output file contains metadata followed by one candidate sequence per line. Edit mode samples substitutions at exactly `mut_k` positions. Duplicate candidates are removed. If five rounds produce fewer than the requested number, the available candidates are saved and a warning is printed. Use `python generate.py --help` for additional options. Input sequences must fit within `max_len` including two special tokens. Architecture arguments must match the checkpoint.

## Authors

Hanwen Sun and Yifan Wu contributed equally, in that order. Corresponding author: Min Li.

School of Information Science and Engineering, Central South University, Changsha, Hunan 410083, China.

Contact: limin@mail.csu.edu.cn
