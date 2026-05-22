"""
Unit tests for the core PPO math: GAE computation and advantage normalization.

These tests don't require any RL environment — they verify the math is correct
against hand-computed reference values and known properties (e.g., GAE with
lambda=1 should equal Monte Carlo returns).

Run:
    pytest tests/ -v
"""
import numpy as np
import pytest
import torch


def compute_gae(rewards, values, dones, next_value, next_done, gamma=0.99, gae_lambda=0.95):
    """Standalone GAE computation, exactly mirroring the loop in ppo.py.

    Args:
        rewards: tensor [T, N]
        values: tensor [T, N]
        dones: tensor [T, N] (done flag *before* step t — i.e. dones[t] means t-1 ended episode)
        next_value: tensor [1, N] — value estimate for obs after the last step
        next_done: tensor [N] — done flag after the last step
    Returns:
        advantages [T, N], returns [T, N]
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(T)):
        if t == T - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t + 1]
            nextvalues = values[t + 1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    returns = advantages + values
    return advantages, returns


class TestGAE:
    """Test GAE math under known conditions."""

    def test_zero_rewards_zero_values_gives_zero_advantage(self):
        T, N = 5, 2
        rewards = torch.zeros(T, N)
        values = torch.zeros(T, N)
        dones = torch.zeros(T, N)
        next_value = torch.zeros(1, N)
        next_done = torch.zeros(N)
        adv, ret = compute_gae(rewards, values, dones, next_value, next_done)
        assert torch.allclose(adv, torch.zeros_like(adv))
        assert torch.allclose(ret, torch.zeros_like(ret))

    def test_lambda_one_equals_monte_carlo_minus_value(self):
        """When lambda=1, GAE advantage = sum of discounted future rewards - V(s_t).

        This is the discounted Monte Carlo return minus the baseline.
        """
        T, N = 4, 1
        rewards = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        values = torch.tensor([[0.0], [0.0], [0.0], [0.0]])
        dones = torch.zeros(T, N)
        next_value = torch.zeros(1, N)
        next_done = torch.zeros(N)
        gamma = 0.9
        adv, ret = compute_gae(rewards, values, dones, next_value, next_done, gamma=gamma, gae_lambda=1.0)

        # With V=0, advantage equals discounted return from each step
        # adv[0] = 1 + 0.9*2 + 0.81*3 + 0.729*4
        expected_0 = 1 + 0.9 * 2 + 0.9**2 * 3 + 0.9**3 * 4
        assert pytest.approx(adv[0, 0].item(), abs=1e-5) == expected_0
        # adv[3] = 4
        assert pytest.approx(adv[3, 0].item(), abs=1e-5) == 4.0

    def test_lambda_zero_equals_td_residual(self):
        """When lambda=0, GAE = r_t + gamma * V(s_{t+1}) - V(s_t) (one-step TD)."""
        T, N = 3, 1
        rewards = torch.tensor([[1.0], [2.0], [3.0]])
        values = torch.tensor([[10.0], [20.0], [30.0]])
        dones = torch.zeros(T, N)
        next_value = torch.tensor([[40.0]])
        next_done = torch.zeros(N)
        gamma = 0.99
        adv, _ = compute_gae(rewards, values, dones, next_value, next_done, gamma=gamma, gae_lambda=0.0)

        # adv[t] should equal one-step TD residual
        assert pytest.approx(adv[0, 0].item(), abs=1e-5) == 1.0 + 0.99 * 20.0 - 10.0
        assert pytest.approx(adv[1, 0].item(), abs=1e-5) == 2.0 + 0.99 * 30.0 - 20.0
        assert pytest.approx(adv[2, 0].item(), abs=1e-5) == 3.0 + 0.99 * 40.0 - 30.0

    def test_done_flag_truncates_bootstrap(self):
        """When dones[t+1]=1, the bootstrap term gamma*V(s_{t+1}) should be zeroed.

        This is the meaning of nextnonterminal = 1 - dones[t+1].
        """
        T, N = 3, 1
        rewards = torch.tensor([[1.0], [1.0], [1.0]])
        values = torch.tensor([[5.0], [5.0], [5.0]])
        # Episode ended right before step 1 (so when computing adv[0], the bootstrap to step 1 should be zero)
        dones = torch.tensor([[0.0], [1.0], [0.0]])
        next_value = torch.tensor([[100.0]])  # large to make the difference visible
        next_done = torch.zeros(N)
        gamma, lam = 0.99, 0.95
        adv, _ = compute_gae(rewards, values, dones, next_value, next_done, gamma=gamma, gae_lambda=lam)

        # adv[0]: nextnonterminal uses dones[1]=1, so delta = r[0] + gamma * V[1] * 0 - V[0] = 1 - 5 = -4
        # And lastgaelam carries from adv[1], but multiplied by nextnonterminal=0, so adv[0] = -4
        assert pytest.approx(adv[0, 0].item(), abs=1e-5) == -4.0

    def test_returns_equal_advantages_plus_values(self):
        """[D-5] PPO uses TD(lambda) returns: returns = advantages + values."""
        T, N = 10, 3
        torch.manual_seed(0)
        rewards = torch.randn(T, N)
        values = torch.randn(T, N)
        dones = (torch.rand(T, N) < 0.1).float()
        next_value = torch.randn(1, N)
        next_done = torch.zeros(N)
        adv, ret = compute_gae(rewards, values, dones, next_value, next_done)
        assert torch.allclose(ret, adv + values)

    def test_shape_preserved(self):
        T, N = 7, 4
        rewards = torch.zeros(T, N)
        values = torch.zeros(T, N)
        dones = torch.zeros(T, N)
        next_value = torch.zeros(1, N)
        next_done = torch.zeros(N)
        adv, ret = compute_gae(rewards, values, dones, next_value, next_done)
        assert adv.shape == (T, N)
        assert ret.shape == (T, N)


class TestAdvantageNormalization:
    """[D-7] Per-minibatch advantage normalization should produce
    mean≈0 and std≈1.
    """

    def test_normalized_has_zero_mean(self):
        adv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = (adv - adv.mean()) / (adv.std() + 1e-8)
        assert abs(normalized.mean().item()) < 1e-5

    def test_normalized_has_unit_std(self):
        adv = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        normalized = (adv - adv.mean()) / (adv.std() + 1e-8)
        assert abs(normalized.std().item() - 1.0) < 1e-3

    def test_constant_advantage_doesnt_explode(self):
        """Edge case: if all advantages are equal, std=0; the 1e-8 epsilon
        keeps the normalization from dividing by zero.
        """
        adv = torch.tensor([2.0, 2.0, 2.0, 2.0])
        normalized = (adv - adv.mean()) / (adv.std() + 1e-8)
        # All normalized values should be 0 (since adv - mean = 0)
        assert torch.allclose(normalized, torch.zeros_like(normalized), atol=1e-5)

    def test_normalization_preserves_sign_pattern(self):
        """Positive advantages relative to the mean stay positive after normalization."""
        adv = torch.tensor([-5.0, -1.0, 0.0, 1.0, 5.0])
        normalized = (adv - adv.mean()) / (adv.std() + 1e-8)
        assert normalized[0] < 0
        assert normalized[-1] > 0
        # Order preserved
        assert torch.all(normalized[:-1] < normalized[1:])


class TestPPOClippedObjective:
    """[D-8] Verify the clipped surrogate is computed as max of (-ratio*adv, -clip(ratio)*adv).

    With minimization (i.e., loss = -objective), this is equivalent to taking
    the pessimistic bound on the policy improvement.
    """

    def test_ratio_one_means_no_clipping_effect(self):
        """At the first epoch's first minibatch, new == old policy so ratio = 1,
        and pg_loss1 == pg_loss2 == -advantage.
        """
        ratio = torch.tensor([1.0, 1.0, 1.0])
        adv = torch.tensor([0.5, -0.3, 0.8])
        clip = 0.2
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * torch.clamp(ratio, 1 - clip, 1 + clip)
        pg_loss = torch.max(pg_loss1, pg_loss2)
        assert torch.allclose(pg_loss, -adv)

    def test_clipping_caps_large_positive_advantage(self):
        """For positive advantages, ratio above 1+eps is clipped — bounding
        how much the policy is rewarded for moving in the +adv direction.
        """
        ratio = torch.tensor([2.0])  # way above 1+clip
        adv = torch.tensor([1.0])
        clip = 0.2
        pg_loss1 = -adv * ratio          # = -2.0
        pg_loss2 = -adv * torch.clamp(ratio, 1 - clip, 1 + clip)  # = -1.2
        pg_loss = torch.max(pg_loss1, pg_loss2)  # = -1.2 (less negative)
        # pessimistic = less reward for the policy = larger (less negative) loss
        assert pg_loss.item() == pytest.approx(-1.2)

    def test_clipping_doesnt_cap_negative_advantage_going_down(self):
        """For negative advantages with ratio < 1-eps, clipping kicks in to
        prevent the policy from being punished too hard.
        """
        ratio = torch.tensor([0.5])  # below 1-clip
        adv = torch.tensor([-1.0])
        clip = 0.2
        pg_loss1 = -adv * ratio          # = 0.5
        pg_loss2 = -adv * torch.clamp(ratio, 1 - clip, 1 + clip)  # = 0.8
        pg_loss = torch.max(pg_loss1, pg_loss2)  # = 0.8
        assert pg_loss.item() == pytest.approx(0.8)


class TestApproxKL:
    """[D-12] Verify the two KL estimators discussed in Schulman's blog
    (http://joschu.net/blog/kl-approx.html).
    """

    def test_k1_estimator(self):
        """k1 = -logratio.mean(), biased but simple."""
        logratio = torch.tensor([0.1, -0.2, 0.05])
        ratio = logratio.exp()
        k1 = (-logratio).mean()
        assert pytest.approx(k1.item(), abs=1e-6) == (-0.1 + 0.2 - 0.05) / 3

    def test_k3_estimator_is_nonnegative(self):
        """k3 = ((ratio - 1) - logratio).mean() — unbiased estimator of KL,
        guaranteed >= 0 because exp(x) - 1 >= x for all x.
        """
        torch.manual_seed(42)
        for _ in range(50):
            logratio = torch.randn(100) * 0.3
            ratio = logratio.exp()
            k3 = ((ratio - 1) - logratio).mean()
            assert k3.item() >= -1e-6  # tiny tolerance for fp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
