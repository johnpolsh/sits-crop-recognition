#

import torch
import warnings
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from typing import Callable, Optional, Union


def get_channels_permuted(img: np.ndarray, channels: list[int]) -> np.ndarray:
    assert isinstance(img, np.ndarray), f"Image must be a numpy array, got {type(img)}"
    assert img.ndim in [4, 3], f"Image must be 3D or 4D, got {img.ndim}"
    return img[..., channels]


def get_channels_permuted_tensor(img: torch.Tensor, channels: list[int]) -> torch.Tensor:
    assert isinstance(img, torch.Tensor), f"Image must be a torch tensor, got {type(img)}"
    assert img.ndim in [4, 3], f"Image must be 3D or 4D, got {img.ndim}"
    return img[channels]


def normalize_img(
        img: np.ndarray,
        norm: Optional[tuple[np.ndarray, np.ndarray]] = None,
        epsilon: float = 1e-8
        ) -> np.ndarray:
    assert isinstance(img, np.ndarray), f"Image must be a numpy array, got {type(img)}"
    assert img.ndim in [4, 3], f"Image must be 3D or 4D, got {img.ndim}"
    if norm is None:
        axis = (1, 2) if img.ndim == 4 else (0, 1)
        mx = np.max(img, axis=axis, keepdims=True)
        mn = np.min(img, axis=axis, keepdims=True)
        return np.clip((img - mn) / (mx - mn + epsilon), 0., 1.)
    
    return np.clip((img - norm[0]) / (norm[1] + epsilon), 0., 1.)


def normalize_img_tensor(
        img: torch.Tensor,
        norm: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        epsilon: float = 1e-8
        ) -> torch.Tensor:
    assert isinstance(img, torch.Tensor), f"Image must be a torch tensor, got {type(img)}"
    assert img.ndim in [4, 3], f"Image must be 3D or 4D, got {img.ndim}"
    if norm is None:
        axis = (-2, -1)
        mx = img.amax(dim=axis, keepdim=True)
        mn = img.amin(dim=axis, keepdim=True)
        return torch.clamp((img - mn) / (mx - mn + epsilon), 0., 1.)
    
    return torch.clamp((img - norm[0]) / (norm[1] + epsilon), 0., 1.)


def img_tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if tensor.ndim == 4:
        return tensor.cpu().detach().numpy().transpose(1, 2, 3, 0)
    elif tensor.ndim == 3:
        return tensor.cpu().detach().numpy().transpose(1, 2, 0)
    else:
        raise ValueError(f"Image must be 3D or 4D, got {tensor.ndim}")


def plot_multi_img(
        img: np.ndarray,
        title: Optional[str] = None,
        subtitle: Optional[list[str]] = None,
        base_size: int = 6
        ):
    if np.any((img < 0.) | (img > 1.)):
        warnings.warn("Image values are not normalized", RuntimeWarning)

    if img.ndim == 4:
        num_imgs = img.shape[0]
        fig, ax = plt.subplots(1, num_imgs, figsize=(base_size * img.shape[0], base_size))
        if title is not None:
            fig.suptitle(title)

        for i in range(num_imgs):
            if subtitle is not None:
                ax[i].set_title(subtitle[i])
            ax[i].imshow(img[i])
            ax[i].axis("off")
        
        fig.tight_layout()
    elif img.ndim == 3:
        fig, ax = plt.subplots(figsize=(base_size, base_size))
        ax.imshow(img)
        ax.axis("off")
        if title is not None:
            ax.set_title(title)
    else:
        raise ValueError(f"Image must be 3D or 4D, got {img.ndim}")

    return fig, ax


def plot_multi_img_tensor(
        img: torch.Tensor,
        title: Optional[str] = None,
        subtitle: Optional[list[str]] = None,
        base_size: int = 6
        ):
    img = img_tensor_to_numpy(img)
    return plot_multi_img(img, title, subtitle, base_size)


def make_grid_tensor(
        img: Union[torch.Tensor, list[torch.Tensor]],
        nrow: int = 8,
        padding: int = 2,
        normalize: bool = False,
        value_range: Optional[tuple[float, float]] = None,
        scale_each: bool = False,
        pad_value: int = 0
        ) -> torch.Tensor:
    if isinstance(img, list):
        assert all(w.ndim == 3 for w in img), "Expected all images to be of format CHW"
        img = torch.cat(img, dim=0)
    else:
        assert img.ndim in [3, 4], "Expected image to be a single CHW image or a tensor of format CDHW"
        img = img.transpose(0, 1)
    
    return make_grid(
        img,
        nrow=nrow,
        padding=padding,
        normalize=normalize,
        value_range=value_range,
        scale_each=scale_each,
        pad_value=pad_value
        )


def mask_to_rgb(
        mask: np.ndarray,
        num_classes: int = 0,
        colormap: Callable[..., np.ndarray] = mpl.colormaps["tab20"]
        ) -> np.ndarray:
    assert mask.ndim == 2, f"Mask must be 2D, got {mask.ndim}"
    if num_classes == 0:
        num_classes = mask.max() + 1
    mask = mask / (num_classes - 1)
    mask = colormap(mask)[..., :3]
    return mask


def mask_tensor_to_rgb_tensor(
        mask: torch.Tensor,
        num_classes: int = 0,
        colormap: Callable[..., np.ndarray] = mpl.colormaps["tab20"]
        ) -> torch.Tensor:
    assert mask.ndim == 2, f"Mask must be 2D, got {mask.ndim}"
    mask = mask.numpy()
    if num_classes == 0:
        num_classes = mask.max() + 1
    mask = mask / (num_classes - 1)
    mask = colormap(mask)[..., :3]
    mask = torch.from_numpy(mask).permute(2, 0, 1)
    return mask
