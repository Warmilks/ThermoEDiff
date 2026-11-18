import os
import glob
import random
import numpy as np
import torch
import pyarrow.parquet as pq
from datetime import timedelta


import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import get_worker_info

from thermoediff.data.collate import HuggingfaceCollaterFunc
from thermoediff.models.diffusion import AlignDiff
from thermoediff.training.trainer import SequenceDiff
from thermoediff.data.stream_dataset import ParquetStreamPairs


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ddp_setup():


    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://",timeout=timedelta(hours=5))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return True, local_rank, rank, world_size
    else:

        return False, 0, 0, 1


def ddp_cleanup(ddp_enabled: bool):
    if ddp_enabled and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(ddp_enabled: bool, rank: int) -> bool:
    return (not ddp_enabled) or (rank == 0)


def estimate_num_rows(shard_glob: str) -> int:
    files = sorted(glob.glob(shard_glob))
    if not files:
        raise FileNotFoundError(f"No parquet shards matched: {shard_glob}")

    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


import random

class DDPParquetStreamPairs(ParquetStreamPairs):
    def __iter__(self):

        rank, world_size = 0, 1
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()


        wi = get_worker_info()
        worker_id, num_workers = (wi.id, wi.num_workers) if wi is not None else (0, 1)

        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers


        my_files = [f for file_idx, f in enumerate(self.files)
                    if (file_idx % global_num_workers) == global_worker_id]


        if not my_files:
            return iter(())


        rng = random.Random(self.seed + 10007 * global_worker_id)

        def gen_once():
            for f in my_files:
                pf = pq.ParquetFile(f)
                for batch in pf.iter_batches(batch_size=self.batch_rows,
                                           columns=["wt_seq", "mut_seq", "dir_vec"]):
                    wts = batch.column("wt_seq").to_pylist()
                    muts = batch.column("mut_seq").to_pylist()
                    vecs = batch.column("dir_vec").to_pylist()
                    for wt, mut, v in zip(wts, muts, vecs):
                        if self._in_split(wt):
                            yield {"seqA": wt, "seqB": mut, "y": v}

        def gen_train_infinite():
            while True:
                rng.shuffle(my_files)
                yield from gen_once()


        return gen_train_infinite() if (self.split == "train") else gen_once()


if __name__ == "__main__":

    ddp_enabled, local_rank, rank, world_size = ddp_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")


    SEED = 9527
    set_seed(SEED + rank)


    shards = "data/l2t_train/train_*.parquet"
    split_ratio = 0.9


    bertDir = "facebook/esm2_t33_650M_UR50D"
    seqMaxLen = 512


    bs_per_gpu = 64
    epochs = 8
    wd = 0.001
    lr = 4e-4
    warmupRatio = 0.02
    earlyStop = 10**9


    dataLoadNumWorkers = 2
    prefetchFactor = 4
    pinMemory = True


    trainDS = DDPParquetStreamPairs(
        shards, split="train", split_ratio=split_ratio, seed=SEED, batch_rows=2048
    )

    validDS = DDPParquetStreamPairs(
    shards, split="valid", split_ratio=split_ratio, seed=SEED, batch_rows=2048
)


    if is_main_process(ddp_enabled, rank):
        N_total = estimate_num_rows(shards)
    else:
        N_total = 0

    if ddp_enabled:
        t = torch.tensor([N_total], device=device, dtype=torch.long)
        dist.broadcast(t, src=0)
        N_total = int(t.item())

    N_train = int(N_total * split_ratio)
    global_bs = bs_per_gpu * world_size
    steps_per_epoch = max(1, N_train // global_bs)

    evalSteps = steps_per_epoch

    maxSteps = evalSteps * epochs
    warmupSteps = int(maxSteps * warmupRatio)

    if is_main_process(ddp_enabled, rank):
        print(f"[INFO] ddp_enabled={ddp_enabled}, world_size={world_size}")
        print(f"[INFO] shards={shards}")
        print(f"[INFO] N_total≈{N_total:,}, N_train≈{N_train:,}")
        print(f"[INFO] bs_per_gpu={bs_per_gpu}, global_bs={global_bs}")
        print(f"[INFO] steps/epoch≈{steps_per_epoch:,}, epochs={epochs}")
        print(f"[INFO] evalSteps={evalSteps:,}, maxSteps={maxSteps:,}, warmupSteps={warmupSteps:,}")


    collater = HuggingfaceCollaterFunc(bertDir, seqMaxLen=seqMaxLen)

    backbone = AlignDiff(
        bertDir=bertDir,
        inSize=1280,
        tknNum=33,
        L=2, H=4, A=2,
        fcSize=1024,
        std=0.27595767
    ).to(device)


    for p in backbone.embedder.parameters():
        p.requires_grad = False

    if ddp_enabled:
        backbone = DDP(
            backbone,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False
        )

    optimizer = torch.optim.AdamW(
        params=filter(lambda p: p.requires_grad, backbone.parameters()),
        weight_decay=wd,
        lr=lr
    )

    AMP = True

    try:
        model = SequenceDiff(model=backbone, optimizer=optimizer, collateFunc=collater, AMP=AMP, DDP=ddp_enabled)
    except TypeError:
        model = SequenceDiff(model=backbone, optimizer=optimizer, collateFunc=collater, AMP=AMP)
        model.DDP = ddp_enabled


    savePath = "ThermoEDiff"
    model.train(
        trainDataSet=trainDS,
        validDataSet=validDS,
        batchSize=bs_per_gpu,
        maxSteps=maxSteps,
        evalSteps=evalSteps,
        earlyStop=earlyStop,
        saveSteps=-1,
        isHigherBetter=True,
        metrics="ACC",
        report=["ACC"],
        savePath=savePath,
        dataLoadNumWorkers=dataLoadNumWorkers,
        prefetchFactor=prefetchFactor,
        pinMemory=pinMemory,
        doEvalTrain=False,
        warmupSteps=warmupSteps
    )

    if ddp_enabled:
        dist.barrier()
    ddp_cleanup(ddp_enabled)
