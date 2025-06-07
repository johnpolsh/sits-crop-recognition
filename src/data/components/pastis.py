#

import json
import geopandas as gpd
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from typing import (
    Any,
    Callable,
    Iterator,
    Literal
)
from tqdm import tqdm
from ..functional import normalize
from ...utils.pylogger import RankedLogger


_logger = RankedLogger(__name__, rank_zero_only=True)


_PROPS = {
    "img_size": (128, 128),
    "num_classes": 20,
    "num_channels": 10,
    "folds": [1, 2, 3, 4, 5],
    "weight": [
        1.0000, 2.1824, 5.8185, 4.6089, 19.6172,
        22.5533, 62.4575, 40.2160, 14.3466, 48.9725,
        54.0988, 37.8966, 42.6151, 142.5403, 25.0291,
        37.8696, 38.6029, 76.8348, 95.2738,  4.1428
    ],
    "class_weight": [
        1.0000, 2.1824, 5.8185, 4.6089, 19.6172,
        22.5533, 62.4575, 40.2160, 14.3466, 48.9725,
        54.0988, 37.8966, 42.6151, 142.5403, 25.0291,
        37.8696, 38.6029, 76.8348, 95.2738
    ],
}
_FOLDS = list[Literal[1, 2, 3, 4, 5]]
_SUBPATCHING_MODE = Literal["sequential", "stride", "equidistant", "random"]


def get_props(props: str | list[str] | None = None) -> dict:
    if props is None:
        return _PROPS
    elif isinstance(props, str):
        return _PROPS[props]
    
    return {prop: _PROPS[prop] for prop in props}


def _get_npy_file_header(file_path: str | Path) -> tuple:
    with open(file_path, 'rb') as file:
        version = np.lib.format.read_magic(file)
        header_info = np.lib.format._read_array_header(file, version) # type: ignore
    return header_info


def _get_npy_file_shape(file_path: str | Path) -> tuple[int, ...]:
    return _get_npy_file_header(file_path)[0]


def _assert_folds_in_range(folds: list):
    assert all(x in _PROPS["folds"] for x in folds),\
        f"Folds must be in {_PROPS['folds']}"


def load_metadata(
        data_dir: str | Path,
        folds: _FOLDS = _PROPS["folds"]
        ) -> gpd.GeoDataFrame:
    _assert_folds_in_range(folds)

    metadata_file_path = Path(data_dir) / "metadata.geojson"
    _logger.info(f"Loading PASTIS metadata from {metadata_file_path}, folds: {folds}")
    metadata_df = gpd.read_file(metadata_file_path)
    metadata_df = metadata_df[metadata_df["Fold"].isin(folds)]
    metadata_df = metadata_df.reset_index(drop=True)
    return metadata_df


def load_data_mean_std(
        data_dir: str | Path,
        folds: _FOLDS = _PROPS["folds"]
        ) -> tuple[np.ndarray, np.ndarray]:
    _assert_folds_in_range(folds)
    
    norm_file_path = Path(data_dir) / "NORM_S2_patch.json"
    _logger.info(f"Loading PASTIS mean and std from {norm_file_path}, folds: {folds}")
    with open(norm_file_path, "r") as f:
        norm_dict = json.load(f)

        mean = [norm_dict[f"Fold_{fold}"]["mean"] for fold in folds]
        mean = np.stack(mean).mean(axis=0).astype(np.float32)

        std = [norm_dict[f"Fold_{fold}"]["std"] for fold in folds]
        std = np.stack(std).mean(axis=0).astype(np.float32)

    return (mean, std)


_metadata = _FOLDS | gpd.GeoDataFrame | Callable[..., gpd.GeoDataFrame]
class PASTISDatasetS2(Dataset):
    PROPS = _PROPS

    def __init__(
            self,
            data_dir: str | Path,
            metadata: _metadata = _PROPS["folds"],
            with_datetime: bool = True,
            normalize: bool = True,
            transform: Callable[[dict], Any] | None = None
            ):
        self.data_dir = Path(data_dir)
        self.with_datetime = with_datetime
        self.normalize = normalize
        self.transform = transform

        if callable(metadata):
            self.metadata = metadata()
        elif isinstance(metadata, list):
            self.metadata = load_metadata(data_dir, metadata)
        elif isinstance(metadata, gpd.GeoDataFrame):
            self.metadata = metadata
        else:
            raise ValueError(f"Invalid metadata type: {type(metadata)}")
        
        if normalize and isinstance(metadata, list):
            self.norm = load_data_mean_std(data_dir, metadata)
        else:
            self.norm = self._calculate_norm_mean_std()

        _logger.info(f"Loaded PASTIS dataset with {len(self.metadata)} samples")

    def _load_patch_data(self, patch_id: int) -> np.ndarray:
        patch_file_path = self.data_dir / "DATA_S2" / f"S2_{patch_id}.npy"
        patch_data = np.load(patch_file_path)
        return patch_data

    def _load_segmentation_annotation(self, patch_id: int) -> np.ndarray:
        target_file_path = self.data_dir / "ANNOTATIONS" / f"TARGET_{patch_id}.npy"
        target = np.load(target_file_path)
        return target

    def _calculate_norm_mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        mean = []
        std = []
        _logger.info("Calculating normalization mean and std")
        for patch_id in tqdm(self.metadata["ID_PATCH"].unique()): # type: ignore
            data = self._load_patch_data(patch_id)
            mean.append(data.mean(axis=(0, 2, 3)))
            std.append(data.std(axis=(0, 2, 3)))
        mean = np.stack(mean).mean(axis=0).astype(np.float32)
        std = np.stack(std).mean(axis=0).astype(np.float32)

        return (mean, std)

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        return normalize(
            data,
            self.norm[0][None, :, None, None],
            self.norm[1][None, :, None, None]
            )

    def _load_dates(self, iloc: int) -> np.ndarray:
        dates = self.metadata.iloc[iloc]["dates-S2"]
        dates = json.loads(dates)
        dates = {k: dates[k] for k in sorted(dates.keys(), key=lambda x: int(x))}
        dates = np.array([
            np.datetime64(
                f"{str(date)[:4]}-{str(date)[4:6]}-{str(date)[6:]}"
                ) for date in dates.values()
            ])
        return dates

    def _iter_targets(self) -> Iterator[np.ndarray]:
        for _, patch in self.metadata.iterrows():
            yield self._load_segmentation_annotation(patch["ID_PATCH"])

    def _iter_data(self) -> Iterator[np.ndarray]:
        for _, patch in self.metadata.iterrows():
            yield self._load_patch_data(patch["ID_PATCH"]).astype(np.float32)

    def _iter_dates(self) -> Iterator[np.ndarray]:
        for _, patch in self.metadata.iterrows():
            yield self._get_dates(patch["ID_PATCH"])

    def _get_target(self, idx: int) -> np.ndarray:
        return self._load_segmentation_annotation(self.metadata.iloc[idx]["ID_PATCH"]).astype(np.int64)
    
    def _get_data(self, idx: int) -> np.ndarray:
        return self._load_patch_data(self.metadata.iloc[idx]["ID_PATCH"]).astype(np.float32)
    
    def _get_dates(self, idx: int) -> np.ndarray:
        return self._load_dates(self.metadata.iloc[idx]["ID_PATCH"])

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> dict:
        patch_id = self.metadata.iloc[idx]["ID_PATCH"]
        data = self._load_patch_data(patch_id).astype(np.float32)
        target = self._load_segmentation_annotation(patch_id).astype(np.int64)

        if self.normalize:
            data = self._normalize_data(data)

        sample = {
            "data": data,
            "target": target,
        }

        if self.with_datetime:
            sample["dates"] = self._load_dates(idx)

        if self.transform:
            sample = self.transform(sample)
        
        return sample


class PASTISSubpatchedDatasetS2(PASTISDatasetS2):
    def __init__(
            self,
            data_dir: str | Path,
            subpatch_size: int,
            metadata: _metadata = _PROPS["folds"],
            subpatching_mode: _SUBPATCHING_MODE = "sequential",
            with_datetime: bool = True,
            normalize: bool = True,
            transform: Callable[[dict], Any] | None = None
            ):
        super().__init__(data_dir, metadata, with_datetime, normalize, transform)
        self.subpatch_size = subpatch_size
        assert subpatching_mode in ["sequential", "stride", "equidistant", "random"],\
            "Subpatching mode must be one of 'sequential', 'stride', 'equidistant' or 'random'"
        self.subpatching_mode = subpatching_mode
        self._gather_subpatches()

    def _gather_subpatches(self):
        columns = self.metadata.columns
        columns = columns.append(pd.Index(["ID_SUBPATCH", "Timestamp_ids"]))
        lines = []
        for _, patch in self.metadata.iterrows():
            patch_id = patch["ID_PATCH"]
            patch_file_path = self.data_dir / "DATA_S2" / f"S2_{patch_id}.npy"
            patch_shape = _get_npy_file_shape(patch_file_path)

            num_subpatches = patch_shape[0] // self.subpatch_size
            remn = patch_shape[0] % self.subpatch_size

            timestamps = np.arange(patch_shape[0])
            if self.subpatching_mode == "sequential":
                timestamps = timestamps[:-remn or None]
            elif self.subpatching_mode == "stride":
                timestamps = timestamps[:-remn or None]
                timestamps = timestamps.reshape(-1, self.subpatch_size).T
            elif self.subpatching_mode == "equidistant":
                timestamps = timestamps[:-remn or None]
                timestamps = timestamps.reshape(self.subpatch_size, -1).T
            elif self.subpatching_mode == "random":
                np.random.shuffle(timestamps)
                timestamps = timestamps[:-remn or None]
            
            timestamps = timestamps.reshape(-1, self.subpatch_size)
            for i in range(num_subpatches):
                subpatch = patch.to_dict()
                subpatch["ID_SUBPATCH"] = i
                subpatch["Timestamp_ids"] = timestamps[i]
                lines.append(subpatch)
        
        self.metadata = gpd.GeoDataFrame(lines, columns=columns)

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> dict:
        subpatch = self.metadata.iloc[idx]
        patch_id = subpatch["ID_PATCH"]
        data = self._load_patch_data(patch_id).astype(np.float32)
        target = self._load_segmentation_annotation(patch_id).astype(np.int64)

        timestamps = subpatch["Timestamp_ids"]
        data = data[timestamps]

        if self.normalize:
            data = self._normalize_data(data)

        sample = {
            "data": data,
            "target": target,
        }

        if self.with_datetime:
            dates = self._load_dates(idx)
            dates = dates[timestamps]
            sample["dates"] = dates

        if self.transform:
            sample = self.transform(sample)
        
        return sample
