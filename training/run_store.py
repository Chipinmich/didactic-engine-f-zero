"""Persistent record of every episode, from every emulator.

Two things need the same data and were going to grow two copies of it:

  the dashboard   wants per-episode summaries, cheaply, while a run is still
                  going -- how far it got, what it hit, where the reward
                  came from

  training        wants the per-step observations and actions, so a later
                  run can learn from an earlier one instead of starting cold
                  every time (training/behaviour_cloning.py)

So both read this. Summaries go in a JSONL file that is appended a line at a
time and is safe to read half-written; the bulky per-step arrays go in one
.npz per episode, written once at the end.

Each emulator writes its OWN summary file (episodes.3.jsonl and so on).
Eight processes appending to one file interleave and corrupt lines sooner or
later, and no amount of flushing fixes that on Windows; readers merge the
files instead, which costs nothing.

    writer = RunWriter(run_id="ppo_v4", instance=3, meta={"algo": "PPO"})
    ...
    for episode in load_episodes("ppo_v4"):
        print(episode["furthest"], episode["reward"])
"""

import glob
import json
import os
import time

import gymnasium as gym
import numpy as np

from bridge_client import TOP_SPEED_RAW

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_DIR = os.path.join(BASE_DIR, "runs", "store")

# Reward is tracked per component by the env, and a flat learning curve is
# usually one term drowning out the others, so the split is worth keeping.
REWARD_PARTS = ("progress", "speed", "wall", "centre", "crash", "lost", "step_cost", "lap")


def run_dir(run_id):
    return os.path.join(STORE_DIR, run_id)


def new_run_id(prefix="run"):
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"


def trajectory_quality(
    off_centre,
    wall_risk,
    curve_signal=None,
    speed=None,
    progress=None,
    frames_in_step=None,
    tail_steps=20,
):
    """Dense, map-independent quality at the difficult end of an episode.

    Checkpoint progress remains the first objective.  This only separates
    genomes that reached the same checkpoint, using visible road geometry:
    clearance, centering, recovery from rising risk, and safe movement near
    the furthest point.  It deliberately contains no checkpoint number or
    turn direction, so the same signal applies to unseen tracks.
    """
    off_centre = np.clip(np.asarray(off_centre, dtype=np.float64), 0.0, 1.0)
    wall_risk = np.clip(np.asarray(wall_risk, dtype=np.float64), 0.0, 1.0)
    count = min(len(off_centre), len(wall_risk))
    if count == 0:
        return {
            "frontier_clearance": 0.0,
            "frontier_centre": 0.0,
            "turn_recovery": 0.0,
            "safe_frontier_rate": 0.0,
            "turn_quality": 0.0,
        }

    off_centre = off_centre[:count]
    wall_risk = wall_risk[:count]
    curve_signal = np.asarray(
        np.zeros(count) if curve_signal is None else curve_signal,
        dtype=np.float64,
    )[:count]
    speed = np.asarray(
        np.zeros(count) if speed is None else speed, dtype=np.float64
    )[:count]
    progress = np.asarray(
        np.arange(count) if progress is None else progress, dtype=np.float64
    )[:count]
    frames = np.asarray(
        np.ones(count) if frames_in_step is None else frames_in_step,
        dtype=np.float64,
    )[:count]
    frames = np.maximum(frames, 1.0)

    tail_start = max(count - max(int(tail_steps), 1), 0)
    tail = slice(tail_start, count)
    tail_weight = frames[tail]
    frontier_clearance = float(
        np.average(1.0 - wall_risk[tail], weights=tail_weight)
    )
    frontier_centre = float(
        np.average(1.0 - off_centre[tail], weights=tail_weight)
    )

    # A falling risk signal during a visible curve is successful correction.
    # Map -0.20..+0.20 risk change to 0..1, with 0.5 meaning no evidence.
    if count > 1:
        change = np.clip(wall_risk[:-1] - wall_risk[1:], -0.20, 0.20)
        active = (
            (np.abs(curve_signal[1:]) >= 0.12)
            | (wall_risk[:-1] >= 0.05)
            | (wall_risk[1:] >= 0.05)
        )
        if active.any():
            mean_change = float(np.average(change[active], weights=frames[1:][active]))
            turn_recovery = float(np.clip(0.5 + mean_change / 0.40, 0.0, 1.0))
        else:
            turn_recovery = 0.5
    else:
        turn_recovery = 0.5

    # Reward safe *moving* frames at the current frontier.  The speed gate
    # prevents a stopped machine from farming this tie-breaker.
    frontier = progress >= (float(np.max(progress)) - 1.0)
    moving = speed >= (0.20 * TOP_SPEED_RAW)
    safe = frontier & moving & (wall_risk < 0.35) & (off_centre < 0.50)
    frontier_frames = float(frames[frontier].sum())
    safe_frontier_rate = (
        float(frames[safe].sum()) / frontier_frames if frontier_frames else 0.0
    )

    turn_quality = float(
        0.40 * frontier_clearance
        + 0.25 * frontier_centre
        + 0.20 * turn_recovery
        + 0.15 * safe_frontier_rate
    )
    return {
        "frontier_clearance": frontier_clearance,
        "frontier_centre": frontier_centre,
        "turn_recovery": turn_recovery,
        "safe_frontier_rate": safe_frontier_rate,
        "turn_quality": turn_quality,
    }


class RunWriter:
    """Records one emulator's episodes into a run.

    Steps are held in memory for the duration of an episode and written once
    it ends. An episode is a few thousand steps of small vectors -- well
    under a megabyte -- and writing per step would put a file write on the
    hot path, which is the one thing the bridge cannot afford.
    """

    def __init__(self, run_id, instance=0, meta=None, keep_steps=True,
                 episode_fields=None):
        self.run_id = run_id
        self.instance = int(instance)
        self.keep_steps = keep_steps
        self.episode_fields = dict(episode_fields or {})

        self.directory = run_dir(run_id)
        self.steps_dir = os.path.join(self.directory, "steps")
        os.makedirs(self.steps_dir, exist_ok=True)

        self.summary_path = os.path.join(
            self.directory, f"episodes.{self.instance}.jsonl"
        )

        # Instance 0 writes the run's metadata; the others would only race to
        # write the same thing.
        if self.instance == 0 and meta is not None:
            with open(os.path.join(self.directory, "meta.json"), "w") as fh:
                json.dump(
                    dict(meta, run_id=run_id, started=time.time()), fh, indent=2
                )

        # A new worker is created for every GA generation (and twice when
        # staged screening is enabled). Continue numbering so step arrays are
        # never overwritten and every summary points at the episode it ran.
        existing = glob.glob(
            os.path.join(self.steps_dir, f"ep_{self.instance}_*.npz")
        )
        numbers = []
        for path in existing:
            try:
                numbers.append(int(os.path.splitext(path)[0].rsplit("_", 1)[1]))
            except (TypeError, ValueError):
                continue
        self.episode = max(numbers, default=-1) + 1
        self._reset_episode()

    def _reset_episode(self):
        self._observations = []
        self._actions = []
        self._rewards = []
        self._speed = []
        self._progress = []
        self._wall = []
        self._racing = []
        self._off_centre = []
        self._wall_risk = []
        self._curve_signal = []
        self._ray_distances = []
        self._safety_override = []
        self._applied_actions = []
        self._applied_steer_counts = []
        self._applied_shoulder_counts = []
        self._frames_in_step = []
        self._agreement_frames = []
        self._started = time.time()

    def record(self, observation, action, reward, info):
        """One policy step. `info` is the env's info dict."""
        self._observations.append(np.asarray(observation, dtype=np.float32))
        self._actions.append(np.asarray(action, dtype=np.int8))
        self._rewards.append(float(reward))
        self._speed.append(int(info.get("speed", 0)))
        self._progress.append(int(info.get("furthest", 0)))
        self._wall.append(bool(info.get("wall", False)))
        self._off_centre.append(float(info.get("off_centre", 0.0)))
        self._wall_risk.append(float(info.get("wall_risk", 0.0)))
        self._curve_signal.append(float(info.get("curve_signal", 0.0)))
        rays = np.asarray(info.get("ray_distances", ()), dtype=np.float32).ravel()
        if rays.size == 0:
            # Keep old/stub environments readable without coupling this data
            # layer to a particular observation width.
            rays = np.zeros(7, dtype=np.float32)
        self._ray_distances.append(rays)
        self._safety_override.append(bool(info.get("safety_override", False)))
        applied_frames = info.get("applied_actions") or []
        if not applied_frames:
            requested = np.asarray(action, dtype=np.int8).ravel()
            if requested.size >= 3:
                applied_frames = [(int(requested[0]), int(requested[2]))]
            else:
                applied_frames = [(int(requested[0]), int(requested[1]))]
        applied = np.asarray(applied_frames, dtype=np.int8).reshape(-1, 2)
        self._applied_actions.append(applied[-1])
        self._applied_steer_counts.append(
            np.bincount(applied[:, 0], minlength=3).astype(np.int16)
        )
        self._applied_shoulder_counts.append(
            np.bincount(applied[:, 1], minlength=3).astype(np.int16)
        )
        self._frames_in_step.append(len(applied))
        requested = np.asarray(action, dtype=np.int8).ravel()
        requested_shoulder = int(requested[1] if requested.size == 2 else requested[2])
        self._agreement_frames.append(
            int(
                np.sum(
                    (applied[:, 0] == int(requested[0]))
                    & (applied[:, 1] == requested_shoulder)
                )
            )
        )

        # Frames where the race is not live: the ~70-frame explosion after
        # the machine is destroyed still reads as racing for a moment (see
        # env/fzero_env.py). Recorded so a training set can drop them rather
        # than silently learning from wreckage.
        self._racing.append(bool(info.get("racing", True)))

    def finish_episode(self, info=None, terminated=False, truncated=False):
        """Close the episode, write it out, and return its summary."""
        info = info or {}
        steps = len(self._rewards)
        if steps == 0:
            return None

        rewards = np.asarray(self._rewards, dtype=np.float32)
        wall = np.asarray(self._wall, dtype=bool)
        speed = np.asarray(self._speed, dtype=np.int32)
        off_centre = np.asarray(self._off_centre, dtype=np.float32)
        wall_risk = np.asarray(self._wall_risk, dtype=np.float32)
        curve_signal = np.asarray(self._curve_signal, dtype=np.float32)
        ray_distances = np.stack(self._ray_distances).astype(np.float32)
        safety_override = np.asarray(self._safety_override, dtype=bool)
        applied_actions = np.stack(self._applied_actions).astype(np.int8)
        applied_steer_counts = np.stack(self._applied_steer_counts).astype(np.int16)
        applied_shoulder_counts = np.stack(self._applied_shoulder_counts).astype(np.int16)
        frames_in_step = np.asarray(self._frames_in_step, dtype=np.int16)
        agreement_frames = np.asarray(self._agreement_frames, dtype=np.int16)
        applied_frames = max(int(frames_in_step.sum()), 1)
        requested_actions = np.stack(self._actions)
        requested_shoulder_column = 1 if requested_actions.shape[1] == 2 else 2
        quality = trajectory_quality(
            off_centre,
            wall_risk,
            curve_signal,
            speed,
            np.asarray(self._progress, dtype=np.int32),
            frames_in_step,
        )
        ray_mid = ray_distances.shape[1] // 2
        left_rays = ray_distances[:, :ray_mid]
        right_rays = ray_distances[:, ray_mid + 1 :]

        summary = {
            **self.episode_fields,
            "episode": self.episode,
            "instance": self.instance,
            "steps": steps,
            "reward": float(rewards.sum()),
            "furthest": int(info.get("furthest", max(self._progress, default=0))),
            "laps": int(info.get("laps_done", 0)),
            "finished": bool(info.get("finished", False)),
            "crashed": bool(info.get("crashed", False)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            # The bottleneck, per the run analysis: contact rate is what
            # drains power, and power is what ends runs.
            "wall_rate": float(wall.mean()),
            "mean_speed": float(speed.mean()),
            "max_speed": int(speed.max()),
            "speed_loss": float(info.get("speed_loss", 0.0)),
            "mean_speed_loss": float(info.get("mean_speed_loss", 0.0)),
            "ray_response_score": float(info.get("ray_response_score", 0.0)),
            "ray_response_opportunities": int(
                info.get("ray_response_opportunities", 0)
            ),
            "ray_toward_wall_rate": float(
                info.get("ray_toward_wall_rate", 0.0)
            ),
            "mean_ray_clearance_recovery": float(
                info.get("mean_ray_clearance_recovery", 0.0)
            ),
            "mean_off_centre": float(
                info.get("mean_off_centre", off_centre.mean())
            ),
            "centre_quality": 1.0
            - float(info.get("mean_off_centre", off_centre.mean())),
            "mean_wall_risk": float(info.get("mean_wall_risk", wall_risk.mean())),
            "safety_overrides": int(info.get("safety_overrides", safety_override.sum())),
            "safety_override_rate": float(
                info.get("safety_override_rate", safety_override.mean())
            ),
            "policy_agreement": float(agreement_frames.sum() / applied_frames),
            "reversal_blocks": int(info.get("reversal_blocks", 0)),
            "actual_lean_rate": float(
                (applied_shoulder_counts[:, 1:].sum()) / applied_frames
            ),
            "requested_lean_rate": float(
                (requested_actions[:, requested_shoulder_column] != 0).mean()
            ),
            "mean_frames_per_step": float(frames_in_step.mean()),
            "wall_hit": bool(info.get("wall_hit", False)),
            "wall_contact_streak": int(info.get("wall_contact_streak", 0)),
            "start_slot": info.get("start_slot"),
            # The per-episode start perturbation (a few opening steer inputs),
            # not just which savestate slot was loaded. Recorded because
            # without it the store cannot answer "is this run actually seeing
            # varied starts, or the same one repeatedly?" -- 8700 episodes
            # were logged before this was added, and every one of them read
            # as an empty variant, which looked exactly like the feature
            # being broken when it was in fact working and simply unlogged.
            "start_variant": list(info.get("start_variant", ()) or ()),
            "start_progress": int(info.get("start_progress", 0)),
            "progress_gain": int(
                info.get(
                    "progress_gain",
                    max(int(info.get("furthest", 0)) - int(info.get("start_progress", 0)), 0),
                )
            ),
            "net_progress": int(
                info.get(
                    "net_progress",
                    int(info.get("progress", 0)) - int(info.get("start_progress", 0)),
                )
            ),
            "episode_frames": int(info.get("episode_frames", frames_in_step.sum())),
            "mean_ray_clearance": float(ray_distances.mean()),
            # The outer rays intentionally look toward the nearby track edge.
            # Use the three forward rays for obstacle clearance so a healthy
            # side boundary does not make every episode appear blocked.
            "closest_ray_clearance": float(
                ray_distances[:, max(0, ray_mid - 1) : ray_mid + 2]
                .min(axis=1)
                .mean()
            ),
            "ray_balance": float(right_rays.mean() - left_rays.mean())
            if left_rays.size and right_rays.size
            else 0.0,
            **quality,
            "seconds": time.time() - self._started,
            "ended": time.time(),
            "reward_parts": {
                key: float(value)
                for key, value in (info.get("reward_parts") or {}).items()
            },
        }

        if self.keep_steps:
            path = os.path.join(
                self.steps_dir, f"ep_{self.instance}_{self.episode:05d}.npz"
            )
            np.savez_compressed(
                path,
                observations=np.stack(self._observations),
                actions=requested_actions,
                rewards=rewards,
                speed=speed,
                progress=np.asarray(self._progress, dtype=np.int32),
                wall=wall,
                racing=np.asarray(self._racing, dtype=bool),
                off_centre=off_centre,
                wall_risk=wall_risk,
                curve_signal=curve_signal,
                ray_distances=ray_distances,
                safety_override=safety_override,
                applied_actions=applied_actions,
                applied_steer_counts=applied_steer_counts,
                applied_shoulder_counts=applied_shoulder_counts,
                frames_in_step=frames_in_step,
                agreement_frames=agreement_frames,
            )
            summary["steps_file"] = os.path.basename(path)

        # Append and flush: the dashboard reads this while the run is going.
        with open(self.summary_path, "a") as fh:
            fh.write(json.dumps(summary) + "\n")
            fh.flush()

        self.episode += 1
        self._reset_episode()
        return summary


def list_runs():
    """Run ids, newest first."""
    if not os.path.isdir(STORE_DIR):
        return []

    runs = [
        name
        for name in os.listdir(STORE_DIR)
        if os.path.isdir(os.path.join(STORE_DIR, name))
    ]
    return sorted(runs, key=lambda name: os.path.getmtime(run_dir(name)), reverse=True)


def load_meta(run_id):
    path = os.path.join(run_dir(run_id), "meta.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def load_episodes(run_id):
    """Every episode across every instance, in the order they finished.

    A half-written final line is skipped rather than raising: this is read
    while training is still appending to it.
    """
    episodes = []
    for path in sorted(glob.glob(os.path.join(run_dir(run_id), "episodes.*.jsonl"))):
        try:
            with open(path) as fh:
                lines = fh.readlines()
        except OSError:
            continue

        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except ValueError:
                # Only ever the last line, and only while it is being
                # written. It will be there next time.
                continue

    episodes.sort(key=lambda e: e.get("ended", 0.0))
    return episodes


def append_generation_summary(run_id, summary):
    """Atomically add or replace one completed GA generation summary."""
    directory = run_dir(run_id)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, "generations.jsonl")
    payload = dict(summary, ended=time.time())
    previous = load_generations(run_id)
    generation = payload.get("generation")
    previous = [item for item in previous if item.get("generation") != generation]
    previous.append(payload)
    previous.sort(key=lambda item: item.get("generation", 0))
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w") as fh:
        for item in previous:
            fh.write(json.dumps(item) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temporary, path)


def load_generations(run_id):
    """Completed generation summaries, tolerating a concurrent partial line."""
    path = os.path.join(run_dir(run_id), "generations.jsonl")
    generations = []
    try:
        with open(path) as fh:
            lines = fh.readlines()
    except OSError:
        return generations
    for line in lines:
        try:
            generations.append(json.loads(line))
        except ValueError:
            continue
    generations.sort(key=lambda item: item.get("generation", 0))
    return generations


def load_steps(run_id, episode):
    """Per-step arrays for one episode summary, or None if not kept."""
    name = episode.get("steps_file")
    if not name:
        return None

    path = os.path.join(run_dir(run_id), "steps", name)
    if not os.path.exists(path):
        return None

    with np.load(path) as data:
        return {key: data[key] for key in data.files}


class RecordingWrapper(gym.Wrapper):
    """Gym wrapper that writes every episode into a run.

    A real gymnasium.Wrapper, which it has to be: stable-baselines3's Monitor
    asserts its argument is one, so training/train.py and train_dqn.py both
    fail at construction otherwise -- and they wrap this in Monitor inside a
    SubprocVecEnv worker, where the assertion surfaces as a BrokenPipeError
    from the parent rather than as anything that names the real problem.
    """

    def __init__(self, env, writer):
        super().__init__(env)
        self.writer = writer
        self._last_info = {}
        self.last_summary = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self._last_info = info
        self.last_summary = None
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.writer.record(observation, action, reward, info)
        self._last_info = info

        if terminated or truncated:
            self.last_summary = self.writer.finish_episode(
                info, terminated, truncated
            )

        return observation, reward, terminated, truncated, info

    def close(self):
        # An episode cut off by the run ending is still worth keeping -- a
        # long run's last episode per emulator would otherwise always be
        # thrown away, and those are the most developed ones.
        self.last_summary = self.writer.finish_episode(
            self._last_info, False, True
        )
        return self.env.close()
