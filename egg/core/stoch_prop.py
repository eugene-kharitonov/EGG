import torch
import torch.distributions
import doctest
import torch.nn as nn
from collections import defaultdict


class ReinforcableMultinomial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        d = torch.distributions.Categorical(logits=logits)
        sample = d.sample()
        losses_storage = torch.zeros(logits.size(0), device=logits.device)
        ctx.save_for_backward(logits, sample, losses_storage)

        one_hot_sample = torch.zeros_like(
            logits).scatter_(-1, sample.unsqueeze(1), 1.0)
        return one_hot_sample

    @staticmethod
    def backward(ctx, _):
        logits, sample, loss = ctx.saved_tensors
        grad = torch.zeros_like(logits).scatter_(-1,
                                                 sample.unsqueeze(1), logits)
        return -loss.unsqueeze(1) * grad


r_multinomial = ReinforcableMultinomial.apply


class StochContext:
    def __init__(self):
        self.baseline_storage = defaultdict(float)
        self.baseline_n = defaultdict(float)

    def propagate_loss(self, **losses):
        for name, loss in losses.items():
            grad_fn = loss.grad_fn
            value = loss.detach()

            baselined_value = value - self.baseline_storage[name]
            self.baseline_n[name] += 1

            assert grad_fn
            nodes = [grad_fn]
            seen = set()

            while nodes:
                node = nodes.pop()
                if 'ReinforcableMultinomial' in str(node.__class__):
                    assert node not in seen
                    *_, losses_storage = node.saved_tensors
                    losses_storage.add_(baselined_value)
                    seen.add(node)
                if node.next_functions:
                    nodes.extend(x for (x, _) in node.next_functions if x)

            self.baseline_storage[name] += (value.mean() -
                                            self.baseline_storage[name]) / self.baseline_n[name]

        return torch.stack([l for l in losses.values()]).sum()
