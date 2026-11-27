# Prompt: make F-Zero RL training faster

Task brief for speeding up training throughput in this repo. Written after
measuring the current run, so the numbers below are observed, not assumed.

## The system

BizHawk emulators run a Lua bridge (`lua/bridge.lua`) in **lockstep**: the
emulator advances exactly one frame per action Python sends. Each frame is
published as a PNG in a memory-mapped file. `bridge_client.py` is the Python
side, `env/fzero_env.py` wraps it as a Gymnasium env, and
`training/train.py` runs PPO across several emulators with SubprocVecEnv.

Actions are held for `frame_skip` emulated frames (default 4). Observations
are 9 track features, stacked 8 deep by `VecFrameStack` into 72.

## Measured baseline

Machine has **32 logical cores**. With 4 emulators, frame-skip 4:

| | |
|---|---|
| Throughput | 32 policy steps/s = **124 emulated frames/s** |
| Total CPU | 87% of 32 cores |
| Each EmuHawk | ~31% of one core (4 total = 1.2 cores) |
| Each env worker | ~18-21% of one core (4 total = 0.8 cores) |
| **PPO learner** | **~1500% = 15 cores** |

Per-frame Python cost, measured separately: PNG decode 0.72ms, feature
extraction 0.055ms. So decoding dominates the vision work by 13x.

## Findings to act on

1. **Torch is using 16 threads for a 72->64->64 MLP.** Benchmarked at the
   real policy size and batch: 16 threads 1.451ms/update, 1 thread
   0.897ms/update. One thread is **1.6x faster** and frees ~15 cores. This
   is the single biggest and cheapest win. `torch.set_num_threads(1)`.

2. **The emulators are nowhere near saturated.** At 31% of a core each they
   are mostly idle, waiting on Python in lockstep. Freeing the learner's
   cores should allow far more instances -- roughly 0.5 core per instance
   including its worker, so a dozen or more should fit.

3. **Every frame is PNG-encoded by BizHawk and decoded by Python, but only
   one frame in `frame_skip` is used as an observation.** Capturing only on
   frames Python will actually look at could remove ~75% of both the encode
   (emulator side) and decode (Python side). The catch: the reward charges
   wall contact per frame, so this changes the reward slightly unless the
   penalty is rescaled -- see the open question below.

4. **Unmeasured, worth checking:** the per-frame file IPC (write action via
   temp+rename, read state file, read mmf) may cost more than expected at
   these rates. Profile before optimising.

## Constraints

- Do not break the reward's tested properties. `tests/test_reward.py` and
  `tests/test_control.py` must pass. In particular: faster laps must pay
  more than slow ones, driving must beat idling, and power must never be
  rewarded.
- Verify throughput changes by measurement, not reasoning. Run before and
  after and report both numbers.
- Beware run-to-run variance: it has been large enough here to fake
  improvements before. Repeat measurements.

## Deliverable

A measured before/after on frames per second, the changes that produced it,
and anything tried that did NOT help (that is as useful as what did).
