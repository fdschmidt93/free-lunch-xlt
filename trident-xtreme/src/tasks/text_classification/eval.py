import torch
import torch.nn.functional as F


def get_preds(self, outputs: dict, *args, **kwargs) -> dict:
    outputs["preds"] = outputs["logits"].argmax(-1)
    return outputs


def get_loss(preds, logits) -> torch.Tensor:
    N = logits.shape[0]
    device = logits.device
    exp_loss = -F.log_softmax(logits, -1)[torch.arange(N, device=device), preds].mean()
    return exp_loss
