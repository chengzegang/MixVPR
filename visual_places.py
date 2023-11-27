import ast
from functools import cached_property, partial
import glob
import os
from typing import List
from datetime import datetime
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
from tqdm.auto import tqdm


def _create_mmap(obj: np.ndarray) -> np.memmap:
    temp = tempfile.NamedTemporaryFile(delete=False)
    arr = np.memmap(
        temp.name,
        dtype=obj.dtype,
        mode="w+",
        shape=obj.shape,
    )
    arr[...] = obj
    arr.flush()
    arr = np.memmap(
        temp.name,
        dtype=obj.dtype,
        mode="r",
        shape=obj.shape,
    )
    return arr


class VisualPlace(Dataset):
    def __init__(self, metadata_desc: str, **kwargs):
        super().__init__()
        self.metadata_desc = metadata_desc
        self._keys = list(kwargs.keys())
        for key, value in kwargs.items():
            setattr(self, key, value)

    @cached_property
    def data(self):
        return np.load(self.metadata_desc, allow_pickle=True).item()

    @classmethod
    def from_visual_place(cls, visual_place: "VisualPlace", **kwargs):
        return cls(
            visual_place.metadata_desc,
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
        coords = np.asarray(coords).astype(np.float32)
        coords = torch.from_numpy(coords)
        ts = np.asarray(ts)
        ts = torch.from_numpy(ts)
        d = cdist(coords, coords)
        sadj = d.as_tensor().le(spatial_radius)
        sadj.fill_diagonal_(False)
        ts = ts.view(-1, 1).float()
        tadj = cdist(ts, ts).as_tensor().le(temporal_radius)
        tadj.fill_diagonal_(False)
        paths = np.asanyarray(paths)
        sadj_ = []
        tadj_ = []

        for i in tqdm(range(len(paths))):
            i_sadj = torch.where(sadj[i])[0].numpy()
            i_tadj = torch.where(tadj[i])[0].numpy()
            sadj_.append(i_sadj)
            tadj_.append(i_tadj)

        data = {
            "path": paths,
            "sadj": sadj_,
            "tadj": tadj_,
        }
        data = np.asarray(data)
        temp = tempfile.NamedTemporaryFile(delete=False)
        np.save(temp, data)
        return cls(
            temp.name,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.data["path"])

    def __getitem__(self, index: int) -> dict:
        path = self.data["path"][index]
        sadj = self.data["sadj"][index]
        tadj = self.data["tadj"][index]
        sadj = [self.data["path"][i] for i in sadj]
        tadj = [self.data["path"][i] for i in tadj]
        return {
            "path": path,
            "sadj": sadj,
            "tadj": tadj,
        }

    def __add__(self, other: "VisualPlace") -> "VisualPlace":
        d1 = self.data
        d2 = other.data.copy()
        offset = len(d1["path"])
        for i in range(len(d2)):
            d2["sadj"][i] = [offset + j for j in d2["sadj"][i]]
            d2["tadj"][i] = [offset + j for j in d2["tadj"][i]]
        paths = np.concatenate([d1["path"], d2["path"]])
        sadj = np.concatenate([d1["sadj"], d2["sadj"]])
        tadj = np.concatenate([d1["tadj"], d2["tadj"]])
        data = {
            "path": paths,
            "sadj": sadj,
            "tadj": tadj,
        }
        data = np.asarray(data)
        temp = tempfile.NamedTemporaryFile(delete=False)
        np.save(temp, data)
        return self.__class__(
            temp.name,
            **self.__dict__.fromkeys(self._keys),
        )

    @property
    def sadj(self) -> MemmapTensor:
        sadj = MemmapTensor(len(self), len(self), dtype=torch.bool, mode="w+")
        for i in range(len(self)):
            i_sadj = self.data["sadj"][i]
            sadj[i, i_sadj] = True
        return sadj

    @property
    def tadj(self) -> MemmapTensor:
        tadj = MemmapTensor(len(self), len(self), dtype=torch.bool, mode="w+")
        for i in range(len(self)):
            i_tadj = self.data["tadj"][i]
            tadj[i, i_tadj] = True
        return tadj

    def _get_stats(self):
        if not hasattr(self, "_stats"):
            avg_sadj = 0
            avg_tadj = 0
            for i in range(len(self)):
                sadj = self.data["sadj"][i]
                tadj = self.data["tadj"][i]
                avg_sadj += len(sadj)
                avg_tadj += len(tadj)
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
        data = None
        if self.data["path"].shape[0] <= size:
            return self
        else:
            index = np.random.choice(len(self), size=size, replace=False)
            paths = self.data["path"][index]
            new_index = dict(zip(paths, range(len(paths))))
            sadj = self.data["sadj"][index]
            tadj = self.data["tadj"][index]
            for i in range(len(sadj)):
                sadj[i] = [new_index[p] for p in sadj[i] if p in new_index]
                tadj[i] = [new_index[p] for p in tadj[i] if p in new_index]
            data = {
                "path": paths,
                "sadj": sadj,
                "tadj": tadj,
            }
            data = np.asarray(data)

            temp = tempfile.NamedTemporaryFile(delete=False)
            np.save(temp, data)
            return self.__class__(
                temp.name,
                **self.__dict__.fromkeys(self._keys),
            )


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
            adj_imgs = [self.load_image(p) for p in adj]
        imgs = [an_img] + adj_imgs
        imgs = torch.stack(imgs)

        return {
            "id": place_id,
            "images": imgs,
        }


def build_nyuvpr360(root: str, **kwargs):
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
        paths, coords, t, spatial_radius=0.00025, temporal_radius=5
    )
    return dataset


def build_nordland(root: str, **kwargs):
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


def build_msls_city(path: str, temporal_radius: float = 5.0, **kwargs):
    postprocessed = os.path.join(path, "database", "postprocessed.csv")
    seq_info = os.path.join(path, "database", "seq_info.csv")
    image_folder = os.path.join(path, "database", "images")
    df_postprocessed = pd.read_csv(postprocessed, index_col=0)
    df_seq_info = pd.read_csv(seq_info, index_col=0)

    spatial_groups = df_postprocessed.groupby("unique_cluster").groups
    temporal_groups = df_seq_info.groupby("sequence_key").groups

    paths = (
        df_postprocessed["key"]
        .apply(lambda x: os.path.join(image_folder, x + ".jpg"))
        .tolist()
    )
    sadj = (
        df_postprocessed["unique_cluster"]
        .apply(lambda x: (spatial_groups[x].to_numpy()).tolist())
        .tolist()
    )
    tadj = df_seq_info["sequence_key"].apply(lambda x: temporal_groups[x])
    frame_number = df_seq_info["frame_number"]

    for i, ta in tadj.items():
        filtered_ta = []
        for t in ta:
            if abs(frame_number[t] - frame_number[i]) <= temporal_radius:
                filtered_ta.append(t)
        tadj[i] = filtered_ta

    index = np.arange(len(paths))
    mapping = dict(zip(df_seq_info.index, index))
    tadj = tadj.apply(lambda x: [mapping[i] for i in x])
    tadj.tolist()
    data = pd.DataFrame({"path": paths, "sadj": sadj, "tadj": tadj})
    data["sadj"] = data["sadj"].apply(lambda x: [data["path"].iloc[i] for i in x])
    data["tadj"] = data["tadj"].apply(lambda x: [data["path"].iloc[i] for i in x])
    data = data.to_numpy()
    data = _create_mmap(data)
    data = VisualPlace(
        {
            "filename": data.filename,
            "shape": data.shape,
            "dtype": data.dtype,
            "mode": data.mode,
        },
        **kwargs,
    )
    return data


def build_msls(root: str, **kwargs):
    datasets = []
    offset = 0
    for city in os.listdir(root):
        city_path = os.path.join(root, city)
        if os.path.isfile(city_path):
            continue
        dataset = build_msls_city(city_path, offset=offset, **kwargs)
        datasets.append(dataset)
        offset += len(dataset)
    dataset = datasets[0]
    for d in datasets[1:]:
        dataset = dataset + d
    return dataset


def build_one_kitti360(root: str, **kwargs):
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


def build_kitti360(roots: List[str], **kwargs):
    if isinstance(roots, str):
        return build_one_kitti360(roots, **kwargs)
    datasets = []
    for root in roots:
        datasets.append(build_one_kitti360(root, **kwargs))
    d0 = datasets[0]
    for d in datasets[1:]:
        d0 += d
    return d0


def get_visual_place(name: str, root: List[str] | str, **kwargs):
    if name == "kitti360":
        return build_kitti360(root, **kwargs)
    elif name == "nyuvpr360":
        return build_nyuvpr360(root, **kwargs)
    elif name == "nordland":
        return build_nordland(root, **kwargs)
    elif name == "msls":
        return build_msls(root, **kwargs)
