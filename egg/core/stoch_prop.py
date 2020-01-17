from typing import Union
import torch
import torch.distributions

import doctest
import torch.nn as nn
from collections import defaultdict


class ReinforcableMultinomial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, probs: torch.Tensor) -> torch.Tensor:
        sample = torch.multinomial(probs, num_samples=1)
        probs_sampled = probs.gather(-1, sample)

        losses_storage = torch.zeros(probs.size(0), device=probs.device)
        ctx.save_for_backward(probs_sampled, sample, losses_storage)

        one_hot_sample = torch.zeros_like(
            probs).scatter_(-1, sample, 1.0)

        return one_hot_sample

    @staticmethod
    def backward(ctx, _x: torch.Tensor) -> torch.Tensor:
        probs, sample, loss = ctx.saved_tensors

        grad = torch.zeros_like(_x)
        for i in range(sample.size(0)):
            grad[i, sample[i]] = 1.0 / probs[i]
        return loss.unsqueeze(1) * grad


r_multinomial: torch.autograd.Function = ReinforcableMultinomial.apply


class ReinforcableCategorical(torch.distributions.Categorical):
    def sample(self) -> torch.Tensor:
        sample_2d = r_multinomial(self.probs)
        return sample_2d


class StochContext(torch.nn.Module):
    def __init__(self, baseline: Union[str, None]=None):
        super().__init__()
        self.baseline = baseline

    def propagate_loss(self, **losses) -> torch.Tensor:
        for name, loss in losses.items():
            grad_fn = loss.grad_fn
            value = loss.detach()

            assert grad_fn
            nodes = [grad_fn]
            seen = set()

            while nodes:
                node = nodes.pop()
                if 'ReinforcableMultinomial' in str(node.__class__):
                    assert node not in seen
                    *_, losses_storage = node.saved_tensors
                    losses_storage.add_(value)
                    seen.add(node)
                if node.next_functions:
                    nodes.extend(x for (x, _) in node.next_functions if x)

        return torch.stack([l for l in losses.values()]).sum()
