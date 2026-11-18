# ThermoEDiff

Implementation of **ThermoEDiff: An Evolution-Direction-Conditioned Diffusion Method for Protein Thermostability Optimization**.

ThermoEDiff uses an evolutionary direction from a reference sequence pair to condition residual diffusion in ESM2 representation space. Fixed-budget decoding generates candidate sequences with a specified number of substitutions.

## Repository structure

```text
ThermoEDiff/
  thermoediff/
    models/       Conditional diffusion model
    data/         Parquet dataset and tokenization
    training/     Training loop, objective and metrics
    inference/    Direction computation and candidate generation
  scripts/        Sequence-pair filtering, alignment and embeddings
  examples/       Example generation plan
  requirements.txt
  README.md
```

## Installation

Use Python 3.10 or newer and a PyTorch build suitable for your CUDA environment.

```bash
pip install -r requirements.txt
```

The encoder is `facebook/esm2_t33_650M_UR50D` from Hugging Face. GPU execution is recommended. Supply your own training data and trained ThermoEDiff checkpoint.

## Data preparation and training

Using a local learn2therm DuckDB database with the `pairs`, `proteins` and `taxa` tables:

```bash
python scripts/prepare_pairs.py --ddb /path/to/learn2therm.ddb --out data/pairs_filtered.parquet
python scripts/align_and_embed.py --in_parquet data/pairs_filtered.parquet --out_dir data/l2t_train
python -m thermoediff.training.train
```

The preprocessing scripts filter homologous pairs, perform global alignment, and write Parquet shards containing `wt_seq`, `mut_seq` and 1280-dimensional `dir_vec` columns. Configure the shard pattern and hyperparameters in `thermoediff/training/train.py`.

For multi-GPU training on Linux with CUDA/NCCL:

```bash
torchrun --standalone --nproc_per_node=4 -m thermoediff.training.train
```

Default settings use a batch size of 64 per GPU, eight approximate epochs, learning rate 0.0004 and warmup ratio 0.02. Provide enough populated shards for the configured ranks and data-loader workers.

## Candidate generation

In the input JSON, `A` and `B` form the reference pair, ordered from lower to higher thermostability. `C_list` contains the target sequences. `examples/plan.json` illustrates the format with short demonstration sequences.

```bash
python -m thermoediff.inference.generate --plan examples/plan.json --ckpt /path/to/checkpoint.pkl --mode edit --mut_k 3 --num 10 --steps 128 --max_rounds 5 --temperature 1.0 --out_root generated_sequences
```

Edit mode selects positions by their best alternative-residue logit gain and samples substitutions from the other 19 amino acids. Output files contain metadata followed by one candidate sequence per line. Duplicates are removed. If the round limit is reached, the available candidates are saved and a warning is printed. Input sequences must fit within `max_len`, including two special tokens. Model architecture arguments must match the checkpoint.

```bash
python -m thermoediff.inference.generate --help
```


