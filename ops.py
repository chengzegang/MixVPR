from typing import Iterable
import torch
from tensordict import MemmapTensor
from torch.utils.data import DataLoader
from torch import Tensor


def _new_copy(old_mmap: MemmapTensor, new_size: int) -> MemmapTensor:
    new_mmap = MemmapTensor(
        *new_size,
        dtype=old_mmap.dtype,
        device=old_mmap.device,
        transfer_ownership=old_mmap.transfer_ownership,
    )
    old_shape = old_mmap.shape
    new_mmap[: old_shape[0], : old_shape[1]] = old_mmap
    return new_mmap


def _cdist_collate_fn(batch):
    return torch.stack(batch, dim=0)


def cdist(
    x: Iterable[Tensor],
    y: Iterable[Tensor],
    buff_size: int = 8192,
    chunk_size: int = 8192,
    device: str = "cuda",
) -> Tensor:
    buff_x_len = buff_size
    buff_y_len = buff_size
    x_ptr = 0
    y_ptr = 0
    buff_d = MemmapTensor(
        buff_x_len,
        buff_y_len,
        dtype=torch.float32,
        device=torch.device("cpu"),
        mode="w+",
    )

    x_dl = DataLoader(
        x, batch_size=chunk_size, num_workers=0, collate_fn=_cdist_collate_fn
    )
    y_dl = DataLoader(
        y, batch_size=chunk_size, num_workers=0, collate_fn=_cdist_collate_fn
    )
    for i, (x_chunk, y_chunk) in enumerate(zip(x_dl, y_dl)):
        x_chunk = x_chunk.to(device)
        y_chunk = y_chunk.to(device)
        d = torch.cdist(x_chunk, y_chunk, compute_mode="donot_use_mm_for_euclid_dist")
        d = d.to(buff_d.dtype)
        changed = False
        if x_ptr + x_chunk.shape[0] > buff_x_len:
            buff_x_len *= 2
            changed = True
        if y_ptr + y_chunk.shape[0] > buff_y_len:
            buff_y_len *= 2
            changed = True
        if changed:
            buff_d = _new_copy(buff_d, (buff_x_len, buff_y_len))
        buff_d[
            x_ptr : x_ptr + x_chunk.shape[0], y_ptr : y_ptr + y_chunk.shape[0]
        ] = MemmapTensor.from_tensor(d)
        x_ptr += x_chunk.shape[0]
        y_ptr += y_chunk.shape[0]

    out = MemmapTensor(x_ptr, y_ptr, dtype=buff_d.dtype, device=buff_d.device)
    out[:] = buff_d[:x_ptr, :y_ptr]
    return out
