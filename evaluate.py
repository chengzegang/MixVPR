import os
from typing import Iterable
import torch
from tensordict import MemmapTensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import Tensor, nn
from ops import cdist
from tqdm.auto import tqdm
from visual_places import MixVPRVisualPlace, VisualPlaceImage


def collate_fn(batch):
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "id": torch.tensor([b["id"] for b in batch]),
        "sadj": [b["sadj"] for b in batch],
        "tadj": [b["tadj"] for b in batch],
    }


def extract(
    model: nn.Module, data: VisualPlaceImage, device: str = "cuda"
) -> MemmapTensor:
    model.eval()
    model = model.to(device)
    dl = DataLoader(
        data,
        batch_size=64,
        num_workers=os.getenv("N_WORKERS", 16),
        collate_fn=collate_fn,
    )

    with torch.no_grad():
        example_feat = model(data[0]["image"][None, ...].to(device))
        feat_size = example_feat.shape[-1]
    features = MemmapTensor(len(data), feat_size, dtype=torch.float32, device="cpu")
    last_i = 0
    with torch.no_grad():
        for batch in tqdm(dl):
            out = model(batch["image"].to(device))
            features[last_i : last_i + out.shape[0]] = out.cpu()
            last_i += out.shape[0]
    return features


def recall(prob: Tensor, gt: Tensor, index: Tensor, ignore_index: int = -1):
    valid = gt.any(dim=-1)

    top10 = torch.topk(prob, 10, dim=-1).indices
    top10_gt = gt.gather(-1, top10)
    top10_index = index.gather(-1, top10)
    top10_gt[top10_index == ignore_index] = False
    top10_gt = top10_gt[valid]
    recall_1 = top10_gt[:, :1].any(dim=-1).float().mean()
    recall_5 = top10_gt[:, :5].any(dim=-1).float().mean()
    recall_10 = top10_gt[:, :10].any(dim=-1).float().mean()
    return recall_1, recall_5, recall_10


def eval_fn(model: nn.Module, dataset: VisualPlaceImage, device: str = "cuda"):
    model.eval()
    model = model.to(device)

    ignore = dataset.tadj.as_tensor()
    ignore.fill_diagonal_(True)
    gt = dataset.sadj.as_tensor()
    valid = gt.any(dim=-1)
    valid = valid.view(-1, 1) & valid.view(1, -1)
    ignore = ignore.masked_fill(~valid, True)

    features = extract(model, dataset, device=device)
    d = cdist(features, features, device=device)
    d = d.masked_fill(ignore, float("inf"))
    prob = torch.softmax(-d, dim=-1)
    index = torch.arange(len(dataset), device=prob.device).view(1, -1).expand_as(gt)
    index = index.masked_fill(ignore, -1)
    recall_1, recall_5, recall_10 = recall(prob, gt, index)
    return {
        "recall_1": recall_1.item(),
        "recall_5": recall_5.item(),
        "recall_10": recall_10.item(),
    }
