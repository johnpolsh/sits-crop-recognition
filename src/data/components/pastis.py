#

import json
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from typing import Any, Callable, Literal, Optional, Union


_PROPS = {
    "img_size": (128, 128),
    "num_classes": 20,
    "num_channels": 10,
    "folds": [1, 2, 3, 4, 5]
}
_FOLDS = list[Literal[1, 2, 3, 4, 5]]
_SUBPATCHING_MODES = Literal["sequential", "stride", "equidistant", "random"]


def _get_npy_file_header(file_path: Union[str, Path]) -> tuple:
    with open(file_path, 'rb') as file:
        version = np.lib.format.read_magic(file)
        header_info = np.lib.format._read_array_header(file, version) # type: ignore
    return header_info


def _get_npy_file_shape(file_path: Union[str, Path]) -> tuple[int, ...]:
    return _get_npy_file_header(file_path)[0]


def _assert_folds_in_range(folds: list):
    assert all(x in _PROPS["folds"] for x in folds),\
        f"Folds must be in {_PROPS['folds']}"


def load_metadata(
        data_dir: Union[str, Path],
        folds: _FOLDS = _PROPS["folds"]
        ) -> gpd.GeoDataFrame:
    _assert_folds_in_range(folds)

    metadata_file_path = Path(data_dir) / "metadata.geojson"
    print(f"Loading PASTIS metadata from {metadata_file_path}, folds: {folds}")
    metadata_df = gpd.read_file(metadata_file_path)
    metadata_df = metadata_df[metadata_df["Fold"].isin(folds)]
    metadata_df = metadata_df.reset_index(drop=True)
    return metadata_df


def load_data_mean_std(
        data_dir: Union[str, Path],
        folds: _FOLDS = _PROPS["folds"]
        ) -> tuple[torch.Tensor, torch.Tensor]:
    _assert_folds_in_range(folds)
    
    norm_file_path = Path(data_dir) / "NORM_S2_patch.json"
    print(f"Loading PASTIS mean and std from {norm_file_path}, folds: {folds}")
    with open(norm_file_path, "r") as f:
        norm_dict = json.load(f)

        mean = [norm_dict[f"Fold_{fold}"]["mean"] for fold in folds]
        mean = np.array(mean).mean(axis=0)
        mean = torch.from_numpy(mean).to(torch.float32)

        std = [norm_dict[f"Fold_{fold}"]["std"] for fold in folds]
        std = np.array(std).mean(axis=0)
        std = torch.from_numpy(std).to(torch.float32)

    return (mean, std)


class PASTISDatasetS2(Dataset):
    PROPS = _PROPS

    def __init__(
            self,
            data_dir: Union[str, Path],
            metadata: Union[_FOLDS, gpd.GeoDataFrame, Callable[..., gpd.GeoDataFrame]] = _PROPS["folds"],
            transform: Optional[Callable[[np.ndarray], Any]] = None,
            target_transform: Optional[Callable[[np.ndarray], Any]] = None,
            class_mapping: Optional[dict] = None
            ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.class_mapping = np.vectorize(lambda x: class_mapping[x]) if class_mapping else None

        if callable(metadata):
            self.metadata = metadata()
        elif isinstance(metadata, list):
            self.metadata = load_metadata(data_dir, metadata)
        elif isinstance(metadata, gpd.GeoDataFrame):
            self.metadata = metadata
        else:
            raise ValueError(f"Invalid metadata type: {type(metadata)}")
        
        print(f"Loaded PASTIS dataset with {len(self.metadata)} samples")

    def _load_patch_data_S2(self, patch_id: int) -> np.ndarray:
        patch_file_path = self.data_dir / "DATA_S2" / f"S2_{patch_id}.npy"
        patch_data = np.load(patch_file_path)
        return patch_data

    def _load_segmentation_annotation(self, patch_id: int) -> np.ndarray:
        target_file_path = self.data_dir / "ANNOTATIONS" / f"TARGET_{patch_id}.npy"
        target = np.load(target_file_path)
        return target

    def _load_data(self, patch_id: int) -> tuple[np.ndarray, np.ndarray]:
        data = (
            self._load_patch_data_S2(patch_id).astype(np.float32),
            self._load_segmentation_annotation(patch_id).astype(np.int64)
            )
        return data

    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        patch_id = self.metadata.iloc[idx]["ID_PATCH"]
        data, target = self._load_data(patch_id)

        if self.class_mapping:
            target = self.class_mapping(target)

        if self.transform:
            data = self.transform(data)

        if self.target_transform:
            target = self.target_transform(target)
        
        return (data, target)


class PASTISSubpatchedDatasetS2(PASTISDatasetS2):
    def __init__(
            self,
            data_dir: Union[str, Path],
            subpatch_size: int,
            metadata: Union[_FOLDS, gpd.GeoDataFrame, Callable[[Union[str, Path]], gpd.GeoDataFrame]] = load_metadata,
            subpatching_mode: _SUBPATCHING_MODES = "equidistant",
            transform: Optional[Callable[[np.ndarray], Any]] = None,
            target_transform: Optional[Callable[[np.ndarray], Any]] = None,
            class_mapping: Optional[dict] = None
            ):
        super().__init__(data_dir, metadata, transform, target_transform)
        self.subpatch_size = subpatch_size
        assert subpatching_mode in ["sequential", "stride", "equidistant", "random"],\
            "Subpatching mode must be one of 'sequential', 'stride', 'equidistant' or 'random'"
        self.subpatching_mode = subpatching_mode
        self.class_mapping = np.vectorize(lambda x: class_mapping[x]) if class_mapping else None
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
        
        self.metadata_df = gpd.GeoDataFrame(lines, columns=columns)

    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        subpatch = self.metadata_df.iloc[idx]
        patch_id = subpatch["ID_PATCH"]
        data, target = self._load_data(patch_id)

        if self.class_mapping:
            target = self.class_mapping(target)

        timestamps = subpatch["Timestamp_ids"]
        data = data[timestamps]

        if self.transform:
            data = self.transform(data)

        if self.target_transform:
            target = self.target_transform(target)
        
        return (data, target)
