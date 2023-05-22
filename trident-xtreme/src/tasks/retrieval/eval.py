import torch

def mrr(scores: torch.Tensor) -> torch.Tensor:
    """Compute MRR from row-aligned matrices of square query-document pairs.

    `mrr` is primarily intended for BLI or sentence-translation retrieval.

    Args:
        scores, torch.Tensor: square matrix of ranking scores

    Returns:
        torch.Tensor: mean reciprocal rank
    """
    N = scores.shape[0]
    rankings = scores.argsort(dim=-1, descending=True) == torch.arange(N, device=scores.device)[:, None]
    reciprocal_rank = 1 / (1 + rankings.float().argmax(dim=-1))
    return reciprocal_rank.mean()
