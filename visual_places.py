import ast
import glob
import os
from typing import List

import jsonlines as jl
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ops import cdist
from torch.utils.data import Dataset
import io
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from torch import Tensor
import torchvision.transforms.functional as TF
from tensordict import MemmapTensor
import tempfile


class VisualPlace(Dataset):
    def __init__(self, data: pd.DataFrame, **kwargs):
        super().__init__()
        self.data = data
        self._keys = list(kwargs.keys())
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_visual_place(cls, visual_place: "VisualPlace", **kwargs):
        return cls(
            visual_place.data,
            **kwargs,
        )

    @classmethod
    def from_metadata(
        cls,
        paths: List[str],
        coords: np.ndarray,
        ts: np.ndarray,
        spatial_radius: float,
        temporal_radius: float,
        **kwargs,
    ):
        coords = np.asarray(coords).astype(np.float64)
        coords = torch.from_numpy(coords)
        ts = np.asarray(ts)
        d = cdist(coords, coords)
        sadj = d <= spatial_radius
        np.fill_diagonal(sadj, False)
        tadj = np.abs(ts[:, None] - ts[None, :]) <= temporal_radius
        np.fill_diagonal(tadj, False)
        buff = io.StringIO()
        with jl.Writer(buff) as writer:
            for i, path in enumerate(paths):
                writer.write(
                    dict(
                        id=i,
                        path=path,
                        sadj=torch.where(sadj[i])[0].tolist(),
                        tadj=torch.where(tadj[i])[0].tolist(),
                    )
                )
        buff.seek(0)
        df = pd.read_json(buff, lines=True)
        return cls(
            df,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, place_id: int) -> dict:
        place = self.data.loc[self.data.id == place_id].iloc[0].to_dict()
        place["sadj"] = [self.data.iloc[i].to_dict() for i in place["sadj"]]
        place["tadj"] = [self.data.iloc[i].to_dict() for i in place["tadj"]]
        print(place)
        return place

    def __add__(self, other: "VisualPlace") -> "VisualPlace":
        df1 = self.data
        df2 = other.data.copy()
        df2["id"] = df2["id"].apply(lambda x: x + len(df1))
        df2["sadj"] = df2["sadj"].apply(lambda x: [i + len(df1) for i in x])
        df2["tadj"] = df2["tadj"].apply(lambda x: [i + len(df1) for i in x])
        df = pd.concat([df1, df2], ignore_index=True)
        return self.__class__(df, **self.__dict__.fromkeys(self._keys))

    def __radd__(self, other: "VisualPlace") -> "VisualPlace":
        return self.__add__(other)

    def __iadd__(self, other: "VisualPlace") -> "VisualPlace":
        return self.__add__(other)

    @property
    def sadj(self) -> MemmapTensor:
        sadj = MemmapTensor(len(self), len(self), dtype=torch.bool, mode="w+")
        for i in range(len(self)):
            row = self.data.iloc[i]
            sadj[row.id, row.sadj] = True
        return sadj

    @property
    def tadj(self) -> MemmapTensor:
        tadj = MemmapTensor(len(self), len(self), dtype=torch.bool, mode="w+")
        for i in range(len(self)):
            row = self.data.iloc[i]
            tadj[row.id, row.tadj] = True
        return tadj

    def _get_stats(self):
        if not hasattr(self, "_stats"):
            avg_sadj = 0
            avg_tadj = 0
            for i in range(len(self)):
                row = self.data.iloc[i]
                avg_sadj += len(row.sadj)
                avg_tadj += len(row.tadj)
            _stats = {
                "total": len(self),
                "avg_sadj": avg_sadj // len(self),
                "avg_tadj": avg_tadj // len(self),
            }
            setattr(self, "_stats", _stats)
        return self._stats

    def __str__(self):
        return (
            "<VisualPlace:"
            + ", ".join([f"{key}:{value}" for key, value in self._get_stats().items()])
            + ">"
        )

    def sample(self, size: int):
        df = self.data.sample(size, ignore_index=True)
        df = df.reset_index()
        ids = df["id"].values
        inds = list(range(len(df)))
        mapping = dict(zip(ids, inds))
        df["sadj"] = df["sadj"].apply(lambda x: [mapping[i] for i in x if i in ids])
        df["tadj"] = df["tadj"].apply(lambda x: [mapping[i] for i in x if i in ids])
        df["id"] = df.index
        return self.__class__(df, **self.__dict__.fromkeys(self._keys))


class VisualPlaceImage(VisualPlace):
    def __getitem__(self, id: int) -> Tensor:
        cluster = super().__getitem__(id)
        image = Image.open(cluster["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if not isinstance(image, Tensor):
            image = TF.to_tensor(image)
        if not image.dtype == torch.float32:
            image = image.to(dtype=torch.float32)
            image = image / 255.0
        return {
            **cluster,
            "image": image,
        }


class MixVPRVisualPlace(VisualPlace):
    def __init__(
        self,
        parquet_path: str,
        img_per_place: int = 4,
        min_img_per_place: int = 4,
        adj_type: str = "sadj",
        **kwargs,
    ):
        super().__init__(parquet_path, **kwargs)
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.adj_type = adj_type

    def load_image(self, path: str) -> np.ndarray:
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __getitem__(self, place_id: int) -> dict:
        cluster = super().__getitem__(place_id)
        an_img = self.load_image(cluster["path"])
        adj = cluster[self.adj_type]
        adj_imgs = None
        if len(adj) == 0:
            adj_imgs = [an_img] * self.img_per_place
        else:
            adj = np.random.choice(
                adj,
                size=self.img_per_place,
                replace=False if len(adj) >= self.min_img_per_place else True,
            )
            adj_imgs = [self.load_image(place["path"]) for place in adj]
        imgs = [an_img] + adj_imgs
        imgs = torch.stack(imgs)

        return {
            **cluster,
            "images": imgs,
        }


def build_nyuvpr360_parquet(root: str, **kwargs):
    import scipy.io as sio

    mat_path = os.path.join(root, "gt_pose.mat")
    mat = sio.loadmat(mat_path)
    coords = mat["pose"].astype(np.float64)
    paths = glob.glob(os.path.join(root, "*.jpg"))
    paths.sort(
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split("_")[1])
    )
    t = np.arange(len(paths))
    dataset = VisualPlace.from_metadata(
        paths, coords, t, spatial_radius=0.0005, temporal_radius=5
    )
    return dataset


def build_nordland_parquet(root: str, **kwargs):
    gt = os.path.join(root, "meta.parquet")
    df = pd.read_parquet(gt)
    df = df.reset_index()
    df["index"] = df.index
    df["path"] = df["path"].apply(lambda x: os.path.join(root, x))
    df["pose"] = df["pose"].apply(lambda x: x.reshape(3, 4))

    df["pose"].iloc[0].shape
    df["coords"] = df["pose"].apply(lambda x: x[:2, 3])
    dataset = VisualPlace.from_metadata(
        df["path"].values,
        np.vstack(df["coords"].to_list()),
        df["index"],
        spatial_radius=4,
        temporal_radius=4,
        **kwargs,
    )
    return dataset


def build_msls_parquet(root: str, **kwargs):
    pass


def build_one_kitti360_parquet(root: str, **kwargs):
    # root = "/mnt/f/datasets/kitti360_scenes/2013_05_28_drive_0000_sync"
    gt = os.path.join(root, "meta.csv")
    df = pd.read_csv(gt)
    df = df.iloc[100:-100:3]
    df = df.reset_index()
    df["index"] = df.index
    df["filename"] = df["filename"].apply(lambda x: os.path.basename(x))
    df["pose"] = df["pose"].apply(
        lambda x: np.asanyarray(ast.literal_eval(x)).reshape(3, 4)
    )
    df["pose"].iloc[0].shape
    df["coords"] = df["pose"].apply(lambda x: x[:2, 3])
    df["path"] = df["filename"].apply(lambda x: os.path.join(root, x))
    dataset = VisualPlace.from_metadata(
        df["path"].values,
        np.vstack(df["coords"].to_list()),
        df["index"],
        spatial_radius=25,
        temporal_radius=5,
        **kwargs,
    )
    return dataset


def build_kitti360_parquet(roots: List[str], **kwargs):
    if isinstance(roots, str):
        return build_one_kitti360_parquet(roots, **kwargs)
    datasets = []
    for root in roots:
        datasets.append(build_one_kitti360_parquet(root, **kwargs))
    d0 = datasets[0]
    for d in datasets[1:]:
        d0 += d
    return d0


def get_visual_place(name: str, root: List[str] | str, **kwargs):
    if name == "kitti360":
        return build_kitti360_parquet(root, **kwargs)
    elif name == "nyuvpr360":
        return build_nyuvpr360_parquet(root, **kwargs)
    elif name == "nordland":
        return build_nordland_parquet(root, **kwargs)
    elif name == "msls":
        return build_msls_parquet(root, **kwargs)
