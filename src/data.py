"""Dataset that turns one frame (and optionally its predecessor) into 40 tiles."""
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment import augment_tile
from .common import DATA
from .imaging import (N_TILES, TILE, TILE_XY, apply_shift, difference_image, gated_shift, load_frame,
                      normalise, tiles)


def load_index() -> pd.DataFrame:
    """frame_index joined with the previous frame of the same sequence and its alignment shift."""
    df = pd.read_csv(DATA / "frame_index.csv").sort_values(["sequence", "timestamp"]).reset_index(drop=True)
    df["label"] = (df["offset_s"] >= 0).astype(int)
    sh = pd.read_csv(DATA / "shifts.csv")[["sequence", "file", "prev_file", "dx", "dy", "phase_resp"]]
    g = [gated_shift(a, b, r) for a, b, r in zip(sh.dx, sh.dy, sh.phase_resp)]
    sh["dx"], sh["dy"] = [a for a, _ in g], [b for _, b in g]
    df = df.merge(sh.drop(columns="phase_resp"), on=["sequence", "file"], how="left")
    df["prev_file"] = df["prev_file"].fillna(df["file"])  # first frame: diff with itself (= no change)
    df[["dx", "dy"]] = df[["dx", "dy"]].fillna(0.0)
    return df


def frame_inputs(row, temporal: bool, align: bool = True):
    """uint8 tiles (40,224,224,3) for the frame; plus difference tiles if temporal."""
    cur = load_frame(row.sequence, row.file)
    if not temporal:
        return tiles(cur), None
    prev = load_frame(row.sequence, row.prev_file)
    if align:
        prev = apply_shift(prev, row.dx, row.dy)
    return tiles(cur), tiles(prev)


class FrameDataset(Dataset):
    def __init__(self, df: pd.DataFrame, temporal: bool, train: bool, align: bool = True, seed: int = 0):
        self.df = df.reset_index(drop=True)
        self.temporal, self.train, self.align = temporal, train, align
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        cur_t, prev_t = frame_inputs(row, self.temporal, self.align)
        rng = np.random.default_rng((self.seed, i, self.epoch))  # new augmentation every epoch, reproducible
        if self.train:
            if self.temporal:
                pairs = [augment_tile(c, rng, p) for c, p in zip(cur_t, prev_t)]
                cur_t = np.stack([a for a, _ in pairs])
                prev_t = np.stack([b for _, b in pairs])
            else:
                cur_t = np.stack([augment_tile(c, rng) for c in cur_t])
        # Return compact uint8 tiles; normalisation and differencing happen on the
        # GPU in model_inputs(). Building 48 MB float32 arrays per sample on the CPU
        # caused millions of page faults per second on Windows and stalled training.
        x = np.ascontiguousarray(cur_t)
        if self.temporal:
            x = np.concatenate([x, np.ascontiguousarray(prev_t)], axis=3)  # (40, 224, 224, 6)
        return torch.from_numpy(x), int(row.label), i


def collate(batch):
    xs, ys, idx = zip(*batch)
    return torch.cat(xs), torch.tensor(ys), torch.tensor(idx)


_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def model_inputs(x_u8: torch.Tensor, temporal: bool, dev: str) -> torch.Tensor:
    """uint8 (N, 224, 224, 3 or 6) -> normalised float (N, 3 or 6, 224, 224) on `dev`.

    Same arithmetic as imaging.normalise / imaging.difference_image on the CPU:
    diff = clip(128 + 2 * (cur - prev), 0, 255), then (v/255 - mean) / std.
    """
    # .contiguous() matters: without it the permuted tensor has channels-last
    # strides, PyTorch propagates that memory layout through the whole network,
    # and training steps ran several times slower on this GPU.
    x = x_u8.to(dev, non_blocking=True).permute(0, 3, 1, 2).float().contiguous()
    mean, std = _MEAN.to(dev), _STD.to(dev)
    cur = (x[:, :3] / 255 - mean) / std
    if not temporal:
        return cur
    d = torch.clamp(128 + 2 * (x[:, :3] - x[:, 3:]), 0, 255)
    return torch.cat([cur, (d / 255 - mean) / std], dim=1)


class ThreadLoader:
    """Batches from a dataset using a thread pool instead of worker processes.

    Why: on Windows, PyTorch's multi-process DataLoader passes batches through
    shared-memory events that failed here ("Couldn't open shared event",
    and earlier a silent hang). JPEG decoding and OpenCV release Python's GIL,
    so threads give most of the speed-up without inter-process memory.
    Prefetches `prefetch` batches ahead.
    """

    def __init__(self, ds, batch_size, shuffle=False, seed=0, threads=8, drop_last=False, prefetch=3):
        self.ds, self.bs, self.shuffle, self.drop_last = ds, batch_size, shuffle, drop_last
        self.threads, self.prefetch = threads, prefetch
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        n = len(self.ds)
        return n // self.bs if self.drop_last else -(-n // self.bs)

    def __iter__(self):
        from concurrent.futures import ThreadPoolExecutor

        order = self.rng.permutation(len(self.ds)) if self.shuffle else np.arange(len(self.ds))
        batches = [order[i:i + self.bs] for i in range(0, len(order), self.bs)]
        if self.drop_last and batches and len(batches[-1]) < self.bs:
            batches = batches[:-1]
        # Each loader thread does its own OpenCV work; letting OpenCV also start its
        # own internal thread pool per call oversubscribes the CPU badly.
        cv2.setNumThreads(1)
        with ThreadPoolExecutor(self.threads) as ex:
            futs = []
            for b in batches[: self.prefetch]:
                futs.append([ex.submit(self.ds.__getitem__, int(i)) for i in b])
            for k in range(len(batches)):
                if k + self.prefetch < len(batches):
                    futs.append([ex.submit(self.ds.__getitem__, int(i)) for i in batches[k + self.prefetch]])
                yield collate([f.result() for f in futs[k]])
                futs[k] = None


__all__ = ["FrameDataset", "load_index", "frame_inputs", "collate", "N_TILES", "TILE", "TILE_XY"]
