from egg.core.stoch_prop import StochContext, r_multinomial

import torch
import torch.distributions
import doctest
import torch.nn as nn
from collections import defaultdict


#torch.random.manual_seed(71)
stoch_context = StochContext()

log_probs = torch.zeros(size=(1, 10), requires_grad=True)
#torch.nn.init.constant_(log_probs, 0.)

optimizer = torch.optim.Adam([log_probs], lr=1e-1)
mean_payouts = torch.tensor([[0, 10, 0, 0, 0, 0, 0, 0, 0, 0]]).float()
#mean_payouts /= mean_payouts.sum()

for _ in range(100):
    optimizer.zero_grad()
    #randomized_payouts = mean_payouts
    randomized_payouts = torch.randn((8, 10)) * 0.0 +  mean_payouts

    one_hot_actions = r_multinomial(log_probs.log_softmax(dim=-1).expand_as(randomized_payouts))
    loss = (one_hot_actions * randomized_payouts).sum(dim=-1)
    #print(loss, one_hot_actions.argmax(dim=-1))

    stoch_context.propagate_loss(loss=loss).backward()
    optimizer.step()
    
print(log_probs.softmax(dim=-1))
stoch_context.baseline_storage


# TODO: hack backward on a variable!