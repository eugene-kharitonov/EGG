import torch

import doctest
import torch.nn as nn
from collections import defaultdict


class ReinforcableMultinomial(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits):
        d = torch.distributions.Categorical(logits=logits)
        logits = (logits - logits.logsumexp(dim=-1, keepdim=True))#.log_softmax(dim=-1)
        #logits.log_softmax(dim=-1)
        sample = d.sample()
        losses_storage = torch.zeros(logits.size(0), device=logits.device)
        ctx.save_for_backward(logits, sample, losses_storage)

        one_hot_sample = torch.zeros_like(
            logits).scatter_(-1, sample.unsqueeze(1), 1.0)

        #print('sampled-1 logprobs', d.log_prob(sample), d.logits)
        return one_hot_sample

    @staticmethod
    def backward(ctx, _):
        logits, sample, loss = ctx.saved_tensors

        #grad = (-1.0 / logits).unsqueeze(-1).expand(sample.size(0), logits.size(0))
        #grad = torch.zeros_like(logits) 


        # expected batch_size x n_logits
        grad = torch.zeros(sample.size(0), logits.size(1), device=sample.device)
        for i in range(sample.size(0)):
            grad[i, sample[i]] = -1.0 / logits[i, sample[i]]

        print(grad, loss)#.size(), sample.size(), logits.size())
        #return loss.unsqueeze(1) * grad
        return loss.unsqueeze(1) * grad


r_multinomial = ReinforcableMultinomial.apply

# TODO: make it a module
class StochContext:
    def __init__(self, baseline=True):
        self.baseline_storage = None # defaultdict(float)
        self.baseline_n = defaultdict(float)
        self.baseline = baseline

    def propagate_loss(self, **losses):
        for name, loss in losses.items():
            grad_fn = loss.grad_fn
            value = loss.detach()

            if not self.baseline:
                baselined_value = value
            else:
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

            if self.baseline:
                self.baseline_storage[name] += (value.mean() -
                                            self.baseline_storage[name]) / self.baseline_n[name]

        return torch.stack([l for l in losses.values()]).sum()
