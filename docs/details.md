# PPO implementation details — extended notes

This is the long-form companion to the README. It walks each detail with the math and the openai/baselines reference, so you can map any line in our code back to the original.

## Why fidelity matters

Engstrom et al. (2020) and Andrychowicz et al. (2021) both showed that PPO's performance is dominated by *implementation details* rather than the core clipped objective. So "implementing PPO" without the details produces something that often doesn't work and is hard to debug. The goal of this repo is to match the canonical implementation `openai/baselines/ppo2` at commit `ea25b9e` line-for-line in behavior.

## The rollout/learning loop (D-1)

```
envs = VecEnv(num_envs=N)
agent = Agent()
next_obs = envs.reset()
next_done = zeros(N)

for update in range(total_updates):
    data = []
    # ROLLOUT
    for step in range(M):
        action, logprob, value = agent(next_obs)
        next_obs, reward, next_done, _ = envs.step(action)
        data.append([obs, action, reward, done, logprob, value])

    # LEARNING
    advantages, returns = gae(data, next_obs, next_done)
    for epoch in range(K):
        for mb in shuffled_minibatches(data, advantages, returns):
            update_policy_and_value(mb)
```

The two key variables for the rollout→learning transition are `next_obs` and `next_done`. `next_obs` is what the value network bootstraps from at the end of the rollout; `next_done` tells you whether to zero out that bootstrap.

## GAE math (D-5)

Given rewards $r_t$, value estimates $V(s_t)$, and the bootstrap value $V(s_{T})$, GAE computes:

$$\delta_t = r_t + \gamma V(s_{t+1}) (1 - d_{t+1}) - V(s_t)$$

$$\hat{A}_t = \delta_t + \gamma \lambda (1 - d_{t+1}) \hat{A}_{t+1}$$

where $d_{t+1} \in \{0, 1\}$ is the done flag.

Then returns (= TD(λ) value targets) are:

$$\hat{R}_t = \hat{A}_t + V(s_t)$$

Setting $\lambda = 1$ recovers the discounted Monte Carlo return (minus the baseline for the advantage). Setting $\lambda = 0$ recovers the one-step TD residual. The default $\lambda = 0.95$ trades a small amount of bias for a large reduction in variance.

**Quirk**: PPO does *not* distinguish truncated episodes from terminated ones. When `gym` returns `done=True` because the time limit was hit, PPO still treats that as a terminal state and skips the value bootstrap. This is technically wrong (the agent didn't really "die"), but baselines does it this way and we preserve the quirk for fidelity.

## Clipped surrogate objective (D-8)

The vanilla policy gradient objective is:

$$L^{PG}(\theta) = \mathbb{E}_t [\log \pi_\theta(a_t \mid s_t) \hat{A}_t]$$

PPO instead optimizes:

$$L^{CLIP}(\theta) = \mathbb{E}_t [\min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)]$$

where $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$ is the probability ratio.

In code we compute the loss to *minimize* (so the signs flip):

```python
pg_loss1 = -mb_advantages * ratio
pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip, 1 + clip)
pg_loss = torch.max(pg_loss1, pg_loss2).mean()
```

`max` of negatives is `min` of the original — same pessimistic bound, just expressed as a loss.

## Value loss clipping (D-9)

$$L^V = \max\left((V_\theta - V_\text{targ})^2, (\text{clip}(V_\theta, V_\text{old} - \epsilon, V_\text{old} + \epsilon) - V_\text{targ})^2\right)$$

Mirrors the policy clip but for the value head, using the same $\epsilon$.

Note: Engstrom et al. and Andrychowicz et al. both find this *doesn't help* in their ablations. We include it because baselines does — for high-fidelity reproduction. Toggle with `--clip-vloss False` if you want to drop it.

## KL approximations (D-12)

From [Schulman's blog](http://joschu.net/blog/kl-approx.html):

| Estimator | Formula | Bias | Variance |
|---|---|---|---|
| k1 | $-\log r$ | unbiased | high |
| k2 | $\frac{1}{2}(\log r)^2$ | biased | low |
| k3 | $(r - 1) - \log r$ | unbiased | low, $\geq 0$ |

Baselines uses k1 (which is what `old_approx_kl` measures). The blog recommends k3 (`approx_kl`) because it has lower variance and is guaranteed non-negative (since $e^x \geq x + 1$). We log both.

## Orthogonal init scales (D-2)

The scale you use determines how the variance propagates through the network at init:

- Hidden layers: `std=sqrt(2)` — matches ReLU/Tanh's variance-preserving property.
- Policy head: `std=0.01` — makes initial action logits near-zero so the policy is near-uniform. Crucial for exploration in early training.
- Value head: `std=1.0` — value outputs are unbounded, no reason to start small.

Why orthogonal? Random orthogonal matrices preserve norms exactly. For deep nets this prevents the kind of gradient vanishing/exploding you can get with Xavier init plus particular activations.

## State-independent log_std (C-2)

In continuous control, the action distribution is $\mathcal{N}(\mu_\theta(s), \sigma)$ — mean depends on the state through the policy network, but $\sigma$ is a separate parameter that doesn't see $s$. We store $\log \sigma$ (not $\sigma$) so it can be unconstrained, and initialize it to 0 so $\sigma = 1$ at the start (full exploration noise).

State-dependent $\sigma$ (as in SAC) is theoretically more expressive but Andrychowicz et al. found it didn't beat the simpler state-independent version on MuJoCo.

## Action clipping vs storage (C-5)

```python
action, logprob, _, value = agent.get_action_and_value(obs)  # unclipped from Normal
clipped_action = clip(action, env.low, env.high)
next_obs, reward, ... = env.step(clipped_action)
# But store the UNCLIPPED action for the PPO update!
actions[step] = action  # not clipped_action
```

Why? The log-prob you computed is the log-prob of the sample from $\mathcal{N}(\mu, \sigma)$, *not* of the clipped action. If you stored the clipped action and later recomputed log-probs, you'd get inconsistent results because the Normal density at a clipped boundary doesn't equal the Normal density at the actual sample.

Fujita et al. (2018) point out this introduces a bias (the clipping changes the effective action distribution) and propose corrections, but baselines just lives with it.

## Frame stacking and image scaling (A-7, A-9)

For Atari, the observation passed to the network is `(4, 84, 84)` — 4 stacked grayscale frames. The 4-frame stack lets the convolutional policy infer velocity and direction of moving objects (a single frame doesn't tell you which way the ball is going).

The pixel values are `uint8` in `[0, 255]`. We divide by 255 inside the network's forward pass (not in the wrapper), which saves memory in the replay buffer / rollout storage — those tensors stay `uint8` until the GPU.

Skipping the divide-by-255 step causes the very first policy update to produce a huge KL divergence because the orthogonal init assumes inputs around unit norm.

## Why this matters for your work

If you're doing adversarial RL or studying alignment in RL agents, the implementation details matter doubly:

- Differences in "clean" performance between methods can vanish or invert when you control for them. Engstrom et al. show this explicitly for PPO vs TRPO.
- Adversarial robustness measurements are sensitive to the exact training distribution induced by these details — reward clipping, observation normalization, etc.
- An adversarial wrapper around a "good" PPO implementation is much more useful as a baseline than one wrapped around a fast-but-buggy implementation that doesn't reach the canonical performance.

This is part of why papers like Andrychowicz et al. (2021) and Engstrom et al. (2020) spent significant effort on ablations rather than novel methods.
