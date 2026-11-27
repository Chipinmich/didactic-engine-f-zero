"""Drive a saved model and report how far it actually gets.

Agreement with a teacher is not driving. A clone can match 84% of the
autopilot's steering inputs and still put the machine into a wall, because
the 16% it misses are not spread evenly -- they cluster exactly where the
decision was hard. So run it.

    python tools/launch_instances.py 8
    python tools/evaluate.py --model bc_ppo --episodes 3
    python tools/evaluate.py --dqn dqn_smoke --episodes 3
    python tools/evaluate.py --random --episodes 3      the honest baseline

Episodes go into the run store like any other, so tools/dashboard.py shows
them and they can be compared later.

Reported against the numbers that matter here: furthest checkpoint (the
autopilot manages about 83) and wall-contact rate (about 40% for the
autopilot, and contact is what drains power and ends runs).
"""

import argparse
import os
import statistics
import sys
import threading
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bridge_client import BridgeTimeout
from env.fzero_env import DEFAULT_FRAME_SKIP, FZeroEnv
from training.run_store import RecordingWrapper, RunWriter, new_run_id
from vision.road_edges import NUM_FEATURES

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
DEFAULT_FRAME_STACK = 8


class Stacker:
    """The frame stack the policy was trained behind, for one env.

    VecFrameStack cannot be used here: it wraps a vectorised env, and this
    drives one emulator per thread. Same layout though -- oldest first,
    newest last, zeros before the episode started -- which is what
    tests/test_run_store.py pins down.
    """

    def __init__(self, stack, features=NUM_FEATURES):
        self.stack = stack
        self.features = features
        self.buffer = np.zeros(stack * features, dtype=np.float32)

    def reset(self, observation):
        self.buffer[:] = 0.0
        self.buffer[-self.features :] = observation
        return self.buffer.copy()

    def push(self, observation):
        self.buffer[: -self.features] = self.buffer[self.features :]
        self.buffer[-self.features :] = observation
        return self.buffer.copy()


def run_one(instance, model, dqn, neuro, episodes, frame_skip, stack, run_id, results,
           rng, assist_mode="full"):
    environment = FZeroEnv(instance=instance, frame_skip=frame_skip, assist_mode=assist_mode)
    writer = RunWriter(
        run_id,
        instance=instance,
        meta={
            "source": "evaluate",
            "model": "random" if (model is None and dqn is None and neuro is None) else "policy",
        },
    )
    recorder = RecordingWrapper(environment, writer)
    stacker = Stacker(stack)

    try:
        for _ in range(episodes):
            observation, info = recorder.reset()
            stacked = stacker.reset(observation)

            while True:
                if neuro is not None:
                    code = neuro["act"](
                        neuro["weights"], stacked, neuro["inputs"], neuro["hidden"]
                    )
                    action = (code // 3, code % 3)
                elif dqn is not None:
                    # DQN predicts one flat Discrete(9) code, same mapping
                    # env/flat_action.py uses: steer = code // 3, shoulder =
                    # code % 3. FZeroEnv.step() wants the decoded pair, not
                    # the flat code -- driving it with the raw code silently
                    # sends nonsense actions instead of raising.
                    code, _ = dqn.predict(stacked[None, :], deterministic=True)
                    code = int(code[0])
                    action = (code // 3, code % 3)
                elif model is None:
                    action = environment.action_space.sample()
                else:
                    action, _ = model.predict(stacked[None, :], deterministic=True)
                    action = action[0]

                observation, _, terminated, truncated, info = recorder.step(action)
                stacked = stacker.push(observation)
                if terminated or truncated:
                    break

            results.append(
                {
                    "furthest": info.get("furthest", 0),
                    "wall": info.get("wall_steps", 0),
                    "centre": 1.0 - float(info.get("mean_off_centre", 1.0)),
                    "risk": float(info.get("mean_wall_risk", 0.0)),
                    "assist": float(info.get("safety_override_rate", 0.0)),
                    "speed": float(info.get("speed", 0.0)),
                    "finished": bool(info.get("finished", False)),
                }
            )
    except BridgeTimeout as exc:
        print(f"  instance {instance}: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"  instance {instance}: {type(exc).__name__}: {exc}")
    finally:
        recorder.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=None, help="legacy SB3 PPO model name under models/")
    parser.add_argument("--dqn", default=None, help="SB3 DQN model name under models/")
    parser.add_argument("--neuro", default=None,
                        help="direct genetic policy .npz file, defaulting to models/ when a name is given")
    parser.add_argument("--random", action="store_true",
                        help="sample actions uniformly -- the baseline to beat")
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0,
                        help="first instance number to use -- match "
                             "launch_instances.py's --offset to evaluate "
                             "several models on separate emulators at once")
    parser.add_argument("--episodes", type=int, default=3, help="per emulator")
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--frame-stack", type=int, default=DEFAULT_FRAME_STACK)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--assist-mode", choices=("full", "emergency"), default="full",
                        help="DrivingAssist strictness to drive under -- must "
                             "match what a model actually trained under for "
                             "the safety-intervention rate to mean anything. "
                             "'full' intervenes on predicted walls, lookahead "
                             "turns and centring; 'emergency' only on an "
                             "actual off-road/near-wall emergency.")
    args = parser.parse_args()

    from training.session_lock import EmulatorSessionLock

    with EmulatorSessionLock(
        "evaluate", offset=args.offset, count=max(args.instances, 1)
    ):
        _run(args)


def _run(args):
    selected = sum(bool(value) for value in (args.random, args.model, args.dqn, args.neuro))
    if selected != 1:
        raise SystemExit("give exactly one of --model NAME, --dqn NAME, --neuro FILE, or --random")

    model = None
    dqn = None
    neuro = None
    if args.model:
        import torch
        from stable_baselines3 import PPO

        torch.set_num_threads(1)
        path = os.path.join(MODEL_DIR, args.model)
        if not os.path.exists(path + ".zip"):
            raise SystemExit(f"no model at {path}.zip")
        model = PPO.load(path)
        print(f"loaded {path}.zip")
    elif args.dqn:
        import torch
        from stable_baselines3 import DQN

        torch.set_num_threads(1)
        path = os.path.join(MODEL_DIR, args.dqn)
        if not os.path.exists(path + ".zip"):
            raise SystemExit(f"no model at {path}.zip")
        # DQN, not PPO.load(): the two algorithms save different policy
        # classes, and loading a DQN checkpoint with PPO.load() either
        # raises or silently reconstructs the wrong network.
        dqn = DQN.load(path)
        print(f"loaded {path}.zip")
    elif args.neuro:
        path = args.neuro
        if not os.path.exists(path):
            path = os.path.join(MODEL_DIR, path)
        if not os.path.exists(path):
            raise SystemExit(f"no genetic policy at {path}")
        from training.neuroevolution import act
        with np.load(path) as data:
            neuro = {
                "weights": data["weights"].astype(np.float32),
                "hidden": int(data["hidden"]),
                "inputs": int(data["inputs"]),
                "act": act,
            }
        print(f"loaded {path}")

    label = "random" if args.random else (args.neuro or args.dqn or args.model)
    run_id = args.run_id or new_run_id(f"eval_{label}")
    print(f"Driving {args.episodes} episodes on each of {args.instances} "
          f"emulators -> run '{run_id}'\n")

    started = time.time()
    results = []
    threads = []
    for instance in range(args.offset, args.offset + args.instances):
        thread = threading.Thread(
            target=run_one,
            args=(instance, model, dqn, neuro, args.episodes, args.frame_skip,
                  args.frame_stack, run_id, results, np.random.default_rng(instance),
                  args.assist_mode),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    if not results:
        raise SystemExit("no episodes completed")

    furthest = [r["furthest"] for r in results]
    spread = statistics.stdev(furthest) if len(furthest) > 1 else 0.0

    print(f"{len(results)} episodes in {(time.time() - started) / 60:.1f} minutes\n")
    print(f"  furthest checkpoint : {statistics.mean(furthest):.1f} +-{spread:.1f}"
          f"  (median {statistics.median(furthest):.0f}, best {max(furthest)})")
    print(f"  laps completed      : {sum(1 for r in results if r['finished'])}")
    print(f"  centre quality      : {statistics.mean(r['centre'] for r in results) * 100:.1f}%")
    print(f"  predicted wall risk : {statistics.mean(r['risk'] for r in results) * 100:.1f}%")
    print(f"  safety intervention : {statistics.mean(r['assist'] for r in results) * 100:.1f}%")
    print(f"\nrun id: {run_id}")
    print("Single runs mean little here -- identical configs have scored 81 "
          "and 66. Repeat before believing a difference.")


if __name__ == "__main__":
    main()
