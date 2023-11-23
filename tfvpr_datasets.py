from typing import Collection, Dict, Iterable, List, Set
import torch
from torch.utils.data import Dataset
import numpy as np
import sqlite3
import sqlmodel as sql
from scipy.spatial.distance import cdist


class Place(sql.SQLModel, table=True):
    id: int = sql.Field(primary_key=True, default=None)
    path: str = sql.Field(index=True)
    x: float
    y: float
    t: float | None = None
    sadj: Set[int] = sql.Field(foreign_key="Place.id", default_factory=set)
    tadj: Set[int] = sql.Field(foreign_key="Place.id", default_factory=set)


engine = sql.create_engine("sqlite:///places.db")
Place.metadata.create_all(engine)


def make_database_from_iterable(places: Iterable[Dict]) -> None:
    with sql.Session(engine) as session:
        for place in places:
            session.add(Place(**place))
        session.commit()
        session.flush()


class VisualPlace(Dataset):
    def __init__(self, places: Set[int]):
        super().__init__()
        self.places = places

    def from_metadata(
        self,
        paths: List[str],
        coords: np.ndarray,
        ts: np.ndarray,
        spatial_radius: float,
        temporal_radius: float,
    ):
        with sql.Session(engine) as session:
            for p, coo, t in zip(paths, coords, ts):
                session.add(Place(path=p, x=coo[0], y=coo[1], t=t))
            session.commit()
            session.flush()
        d = cdist(coords, coords)
        sadj = d <= spatial_radius
        np.fill_diagonal(sadj, False)
        tadj = np.abs(ts[:, None] - ts[None, :]) <= temporal_radius
        np.fill_diagonal(tadj, False)
        with sql.Session(engine) as session:
            for i in range(len(paths)):
                place = session.get(Place, i)
                place.sadj = set(np.where(sadj[i])[0])
                place.tadj = set(np.where(tadj[i])[0])
            session.commit()
            session.flush()

    def sample_spatial_neigbhors(
        self, place_id: int, sample_size: int
    ) -> Collection[Place]:
        with sql.Session(engine) as session:
            place = session.get(Place, place_id)
            query = (
                sql.select(Place)
                .where(Place.id in place.sadj and place.id in self.places)
                .order_by(sql.func.random)
                .limit(sample_size)
            )
            result = session.exec(query).fetchall()
            return result

    def sample_temporal_neighors(
        self, place_id: int, sample_size: int
    ) -> Collection[Place]:
        with sql.Session(engine) as session:
            place = session.get(Place, place_id)
            query = (
                sql.select(Place)
                .where(Place.id in place.tadj)
                .order_by(sql.func.random)
                .limit(sample_size)
            )
            result = session.exec(query).fetchall()
            return result

    def __getitem__(self, place_id: int) -> Place:
        sadj = self.sample_spatial_neigbhors(place_id, 10)
        return {
            "place": place_id,
            "sadj": sadj,
        }
