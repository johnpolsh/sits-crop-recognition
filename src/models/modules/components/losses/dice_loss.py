#

import torch
from torch.nn import functional
from torch.nn.modules.loss import _Loss
from typing import Literal, Optional
from ..functional import soft_dice_score


class DiceLoss(_Loss):
    def __init__(
            self,
            mode: Literal["binary", "multiclass", "multilabel"] = "binary",
            classes: Optional[list[int]] = None,
            log_loss: bool = False,
            from_logits: bool = False,
            label_smoothing: float = 0.,
            ignore_index: Optional[int] = None,
            eps: float = 1e-7
            ):
        assert mode in ["binary", "multiclass", "multilabel"],\
            f"Got {mode=}, expected one of ['binary', 'multiclass', 'multilabel']"
        super().__init__()
        self.mode = mode
        self.classes = classes
        if classes is not None:
            assert mode != "binary", "Masking classes is not supported for binary mode"
            self.classes = torch.tensor(classes, dtype=torch.long)
        
        self.log_loss = log_loss
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert logits.shape[0] == target.shape[0],\
            f"Expected {logits.shape[0]=} to be equal to {target.shape[0]=}"
        
        y_pred = logits
        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == "multiclass":
                y_pred = logits.log_softmax(dim=1).exp()
            else:
                y_pred = functional.logsigmoid(logits).exp()
        y_true = target

        B = target.shape[0]
        C = logits.shape[1]
        if self.mode == "binary":
            y_true = y_true.view(B, 1, -1)
            y_pred = y_pred.view(B, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_true = y_true * mask
                y_pred = y_pred * mask

        elif self.mode == "multiclass":
            y_true = y_true.view(B, -1)
            y_pred = y_pred.view(B, C, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask.unsqueeze(1)

                y_true = functional.one_hot((y_true * mask).to(torch.long), C)
                y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)
            else:
                y_true = functional.one_hot(y_true, C)
                y_true = y_true.permute(0, 2, 1)

        elif self.mode == "multilabel":
            y_true = y_true.view(B, C, -1)
            y_pred = y_pred.view(B, C, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_true = y_true * mask
                y_pred = y_pred * mask
        
        scores = soft_dice_score(
            y_pred,
            y_true.type_as(y_pred),
            label_smoothing=self.label_smoothing,
            eps=self.eps,
            dim=(0, 2)
            )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1. - scores
        
        mask = y_true.sum(dim=(0, 2)) > 0.
        loss *= mask.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]
        
        return loss.mean()
