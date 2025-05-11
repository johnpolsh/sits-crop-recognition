#

import torch


def hausdorff_distance(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    assert y_true.shape == y_pred.shape, "Input tensors must have the same shape."

    dist = torch.cdist(y_true, y_pred, p=2)

    # # Flatten the tensors
    # y_true_flat = y_true.view(-1)
    # y_pred_flat = y_pred.view(-1)

    # # Create a mask to ignore the specified index
    # mask = (y_true_flat != ignore_index) & (y_pred_flat != ignore_index)

    # # Compute the Hausdorff distance
    # hausdorff_distance = torch.max(torch.abs(y_true_flat[mask] - y_pred_flat[mask]))

    return dist
