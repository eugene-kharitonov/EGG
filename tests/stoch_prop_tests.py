from egg.core.stoch_prop import StochContext, r_multinomial, ReinforcableCategorical

import torch
import torch.distributions
import torch.nn as nn


def test_multiarm_bandit():
    torch.random.manual_seed(71)
    stoch_context = StochContext()

    log_probs = torch.zeros(size=(1, 10), requires_grad=True)
    optimizer = torch.optim.Adam([log_probs], lr=1e-1)

    mean_payouts = torch.tensor([[10, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).float()

    for _ in range(100):
        optimizer.zero_grad()
        randomized_payouts = torch.randn((8, 10)) * 0.5 + mean_payouts

        one_hot_actions = r_multinomial(
            log_probs.softmax(dim=-1).expand_as(randomized_payouts))
        loss = -(one_hot_actions * randomized_payouts).sum(dim=-1)

        stoch_context.propagate_loss(loss=loss).backward()
        optimizer.step()

    assert log_probs.argmax().eq(0).all()


def test_two_losses():
    torch.random.manual_seed(71)
    stoch_context = StochContext()

    log_probs_1 = torch.zeros(size=(1, 5), requires_grad=True)
    log_probs_2 = torch.zeros(size=(1, 5), requires_grad=True)

    mean_payouts = torch.tensor([[10, 0, 0, 0, 0]]).float()

    loss_1 = (mean_payouts * r_multinomial(log_probs_1.softmax(dim=-1))).sum()
    loss_2 = (mean_payouts * r_multinomial(log_probs_2.softmax(dim=-1))).sum()

    stoch_context.propagate_loss(loss_1=loss_1).backward()
    assert log_probs_2.grad is None
    assert log_probs_1.grad is not None

    old_grad_1 = log_probs_1.grad.clone()
    stoch_context.propagate_loss(loss_2=loss_2).backward()
    assert log_probs_2.grad is not None

    assert (log_probs_1.grad == old_grad_1).all()


def test_is_policy_gradient():
    mean_payouts = torch.randn(5)
    mean_payouts.zero_()
    mean_payouts[4] = 1

    torch.random.manual_seed(72)
    log_probs = torch.randn(1, 5)
    log_probs.requires_grad_(True)

    stoch_context = StochContext()
    d = ReinforcableCategorical(logits=log_probs)

    sample = d.sample()
    loss_1 = (mean_payouts * sample).sum()
    stoch_context.propagate_loss(loss=loss_1).backward()

    stoch_prop_grad = log_probs.grad
    assert stoch_prop_grad is not None

    torch.random.manual_seed(72)
    log_probs = torch.randn(1, 5)
    log_probs.requires_grad_(True)

    distr = torch.distributions.Categorical(logits=log_probs)
    sample = distr.sample()
    sampled_log_prob = distr.log_prob(sample)

    loss_2 = sampled_log_prob * mean_payouts[sample]
    loss_2.backward()

    policy_grad = log_probs.grad

    assert torch.allclose(policy_grad, stoch_prop_grad), str(policy_grad) + ' ' + str(stoch_prop_grad)


if __name__ == '__main__':
    test_is_policy_gradient()
