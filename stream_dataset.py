import glob
import hashlib
import torch
import pyarrow.parquet as pq
from torch.utils.data import get_worker_info

class ParquetStreamPairs(torch.utils.data.IterableDataset):
    def __init__(self, shard_glob: str, split: str = "train", split_ratio: float = 0.9, seed: int = 0, batch_rows: int = 2048):
        super().__init__()
        self.files = sorted(glob.glob(shard_glob))
        assert self.files, f"no shards matched: {shard_glob}"
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.batch_rows = batch_rows

    def _in_split(self, wt_seq: str) -> bool:
        h = hashlib.blake2b((wt_seq + str(self.seed)).encode(), digest_size=8).digest()
        x = int.from_bytes(h, "little") / (2**64)
        return (x < self.split_ratio) if (self.split == "train") else (x >= self.split_ratio)

    def __iter__(self):
        # --- DDP rank/world_size ---
        rank, world_size = 0, 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()

        # --- DataLoader worker_id/num_workers ---
        wi = get_worker_info()
        worker_id, num_workers = (wi.id, wi.num_workers) if wi is not None else (0, 1)

        # 把 (rank, worker) 合并成一个全局 worker 维度，确保不重复读
        global_worker_id = rank * num_workers + worker_id
        global_num_workers = world_size * num_workers

        for file_idx, f in enumerate(self.files):
            # 关键：按文件分片（简单稳妥）。不同 rank/worker 只读自己那一份
            if (file_idx % global_num_workers) != global_worker_id:
                continue

            pf = pq.ParquetFile(f)
            for batch in pf.iter_batches(batch_size=self.batch_rows, columns=["wt_seq", "mut_seq", "dir_vec"]):
                wts = batch.column("wt_seq").to_pylist()
                muts = batch.column("mut_seq").to_pylist()
                vecs = batch.column("dir_vec").to_pylist()
                for wt, mut, v in zip(wts, muts, vecs):
                    if self._in_split(wt):
                        yield {"seqA": wt, "seqB": mut, "y": v}
