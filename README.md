# PPO from Scratch

A faithful, single-file PyTorch reproduction of Proximal Policy Optimization (PPO), following all 37 implementation details documented in [Huang et al., "The 37 Implementation Details of Proximal Policy Optimization" (ICLR Blog Track, 2022)](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/).

Three single-file implementations:

- `ppo/ppo.py` — classic control (CartPole, Acrobot). 13 core details.
- `ppo/ppo_atari.py` — Atari (Breakout, Pong, BeamRider). 13 core + 9 Atari details.
- `ppo/ppo_continuous.py` — continuous control (MuJoCo, Pendulum). 13 core + 9 continuous details.

Each file is self-contained — you can read it top to bottom and see every detail. Implementation details are tagged inline with `[D-N]`, `[A-N]`, `[C-N]` for cross-referencing with this README.

## Quick start

```bash
# install
pip install -r requirements.txt

# classic control
python -m ppo.ppo --env-id CartPole-v1 --total-timesteps 500000

# atari
python -m ppo.ppo_atari --env-id BreakoutNoFrameskip-v4 --total-timesteps 10000000

# continuous control
python -m ppo.ppo_continuous --env-id Hopper-v4 --total-timesteps 1000000

# with wandb tracking
python -m ppo.ppo --track --wandb-project-name ppo-from-scratch
```

Run the tests:

```bash
pytest tests/ -v
```

## The 13 core implementation details

These apply regardless of the task. Tags `[D-N]` appear next to the corresponding code in `ppo/ppo.py`.

**[D-1] Vectorized architecture.** A single learner rolls out `N` parallel environments for `M` steps each, producing an `N*M`-step batch. The loop alternates between a **rollout phase** (collecting trajectories) and a **learning phase** (updating policy from the collected batch). The variables `next_obs` and `next_done` are what bridge the two phases — they let PPO bootstrap the value of the final state and continue the rollout in the next iteration. This is what enables PPO to learn from long-horizon games it never finishes.

**[D-2] Orthogonal init + constant bias init.** Hidden layers use orthogonal initialization with `std=sqrt(2)`. The policy head uses `std=0.01` (so the initial policy is near-uniform and exploration is broad), the value head uses `std=1.0`. All biases are zero. Engstrom et al. (2020) found this beats Xavier init.

**[D-3] Adam epsilon = 1e-5.** PyTorch's default is `1e-8`, TensorFlow's is `1e-7`. Baselines uses `1e-5`. Not in the paper, not documented anywhere — just a quirk you have to match for fidelity.

**[D-4] Linear learning rate annealing.** LR linearly decays from initial value to 0 over training. Atari starts at `2.5e-4`, MuJoCo at `3e-4`.

**[D-5] Generalized Advantage Estimation.** Two sub-details:
- **Value bootstrap**: at the end of a rollout, the value of `next_obs` is used as the bootstrap target. PPO does *not* correctly distinguish truncation from termination — both are treated as terminal. This is a known fidelity quirk preserved here.
- **TD(λ) returns**: `returns = advantages + values`, which equals TD(λ) when computed via GAE. Monte Carlo is the special case `λ=1`.

**[D-6] Mini-batch updates.** Each epoch, the `N*M` flat batch indices are shuffled and split into mini-batches. Common bugs: (a) updating on the whole batch, (b) sampling random indices that don't cover the whole batch.

**[D-7] Advantage normalization at the minibatch level.** Subtract mean, divide by std + 1e-8. Crucially, this is done *per minibatch*, not over the whole batch.

**[D-8] Clipped surrogate objective.** The PPO clipped policy loss:

```
L_clip = max( -A * ratio, -A * clip(ratio, 1-ε, 1+ε) )
```

`max` of negatives = pessimistic bound on policy improvement.

**[D-9] Value function loss clipping.** Symmetric clipping for the value function:

```
L_V = max( (V_new - V_target)^2, (clip(V_new, V_old - ε, V_old + ε) - V_target)^2 )
```

Engstrom et al. and Andrychowicz et al. find this *doesn't* actually help or even hurts — but it's part of the canonical implementation, so we keep it for reproducibility.

**[D-10] Overall loss + entropy bonus.** One optimizer steps a combined loss:

```
loss = policy_loss - entropy_coef * entropy + vf_coef * value_loss
```

Policy and value share the optimizer (and on Atari, share the CNN trunk).

**[D-11] Global gradient clipping.** The concatenated gradient norm is clipped at `0.5` before each optimizer step.

**[D-12] Debug variables.** Track `policy_loss`, `value_loss`, `entropy`, `clipfrac` (fraction of samples that hit the clip), `approx_kl` (Schulman's k1 and k3 estimators from his [KL approximation blog post](http://joschu.net/blog/kl-approx.html)). High KL or unusual clipfrac is the first sign something is wrong.

**[D-13] Shared vs separate networks for policy and value.** The blog shows separate networks beat shared networks on simple envs (the value and policy losses compete in the shared trunk). We default to separate networks in `ppo.py`. The Atari variant uses a shared CNN trunk for compute, see `[A-8]`.

## The 9 Atari-specific details (`ppo/ppo_atari.py`)

**[A-1] NoopResetEnv** — take 1–30 random no-ops on reset to inject stochasticity.

**[A-2] MaxAndSkipEnv** — repeat each action 4 frames; the observation is the per-pixel max of the last 2 frames (handles flickering sprites).

**[A-3] EpisodicLifeEnv** — treat life loss as end-of-episode during training. Note Machado et al. (2018) actually suggest not using this; we follow baselines for fidelity.

**[A-4] FireResetEnv** — automatically press FIRE on reset for games like Breakout. Nobody knows where this wrapper originally came from. Folklore.

**[A-5] WarpFrame** — extract luminance, resize to 84×84.

**[A-6] ClipRewardEnv** — clip reward sign to `{-1, 0, +1}`. Limits the scale of value targets across very different score scales.

**[A-7] FrameStack** — stack 4 consecutive frames so the agent can infer velocity.

**[A-8] Shared Nature-CNN trunk.** The CNN from Mnih et al. (2015) followed by separate linear heads for policy and value. Shared trunk saves compute and was the baselines default.

**[A-9] Scale images to [0,1].** Divide uint8 pixels by 255 *inside* the network. Anecdotally critical — skip it and the first policy update blows up the KL.

## The 9 continuous-control details (`ppo/ppo_continuous.py`)

**[C-1] Normal distribution** for sampling continuous actions.

**[C-2] State-independent log_std**, stored as an `nn.Parameter` initialized to 0. The mean is state-dependent (output by the actor MLP), but the std is a free vector.

**[C-3] Independent action components.** For multi-dim actions, treat each dimension as independent: `log_prob(a) = sum over dims of log_prob(a_d)`.

**[C-4] Separate MLPs for policy and value.** Unlike the Atari shared CNN case, MuJoCo uses fully separate `64-64-Tanh` MLPs. Andrychowicz et al. (2021) found this consistently helps.

**[C-5] Action clipping.** Sampled actions can exceed env bounds, so they're clipped before passing to the simulator — but the *unclipped* action is what's stored and used to compute log-probs. (Stored unclipped action keeps the log-prob calculation consistent with the actual sample from the Normal distribution.)

**[C-6] Observation normalization** via running mean/std (`gym.wrappers.NormalizeObservation`).

**[C-7] Observation clipping** to `[-10, 10]` *after* normalization.

**[C-8] Reward scaling** by the std of a rolling discounted-return estimate (`gym.wrappers.NormalizeReward`). Note: only scaled (divided), not centered.

**[C-9] Reward clipping** to `[-10, 10]` after scaling.

## Debugging tips

When something looks off:

1. **Seed everything**, then print sums of obs / actions / values at the same step in your impl and the reference. Find the first place they diverge.
2. **Check `ratio == 1` on the first minibatch of the first epoch.** If not, you haven't reconstructed the rollout policy correctly.
3. **Watch approx_kl.** Should stay below ~0.02. If it spikes early, something's broken (often `[A-9]`-style scaling missing, or wrong layer init).
4. **The "400 in Breakout" rule.** If your Atari PPO can't get to ~400 episodic return on Breakout in ~10M frames, you're missing details.

## Project layout

```
ppo-from-scratch/
├── ppo/
│   ├── ppo.py              # classic control — 13 core details
│   ├── ppo_atari.py        # + 9 Atari details
│   └── ppo_continuous.py   # + 9 continuous-control details
├── tests/
│   └── test_gae.py         # unit tests for GAE, advantage norm, clipped objective, KL estimators
├── configs/
│   ├── classic.yaml
│   ├── atari.yaml
│   └── mujoco.yaml
├── docs/
│   └── details.md          # detail-by-detail reference
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── README.md
```

## Hyperparameters

Defaults from `openai/baselines/ppo2/defaults.py`. Override on the command line.

| | Classic | Atari | MuJoCo |
|---|---|---|---|
| `num-envs` (N) | 4 | 8 | 1 |
| `num-steps` (M) | 128 | 128 | 2048 |
| `learning-rate` | 2.5e-4 | 2.5e-4 | 3e-4 |
| `num-minibatches` | 4 | 4 | 32 |
| `update-epochs` (K) | 4 | 4 | 10 |
| `clip-coef` (ε) | 0.2 | 0.1 | 0.2 |
| `ent-coef` | 0.01 | 0.01 | 0.0 |
| `gamma` | 0.99 | 0.99 | 0.99 |
| `gae-lambda` | 0.95 | 0.95 | 0.95 |

The Atari wrappers (`NoopResetEnv`, `MaxAndSkipEnv`, `EpisodicLifeEnv`,
`FireResetEnv`, `ClipRewardEnv`) used in `ppo/ppo_atari.py` come from
[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), which in
turn ports them from the original `openai/baselines` repository.

## Citation

If you use this implementation, please also cite the blog post that documents the 37 details:

```bibtex
@misc{huang2022thirty-seven,
  title  = {The 37 Implementation Details of Proximal Policy Optimization},
  author = {Huang, Shengyi and Dossa, Rousslan Fernand Julien and Raffin, Antonin and Kanervisto, Anssi and Wang, Weixun},
  year   = 2022,
  url    = {https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/}
}

@article{schulman2017proximal,
  title   = {Proximal Policy Optimization Algorithms},
  author  = {Schulman, John and Wolski, Filip and Dhariwal, Prafulla and Radford, Alec and Klimov, Oleg},
  journal = {arXiv preprint arXiv:1707.06347},
  year    = 2017
}
@article{stable-baselines3,
  author  = {Raffin, Antonin and Hill, Ashley and Gleave, Adam and Kanervisto, Anssi and Ernestus, Maximilian and Dormann, Noah},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1--8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
```

## License

MIT.
