"""Record the hand-written autopilot driving, as a teacher for the agent.

PPO from a random start has never learned anything here: the longest run was
73k steps with the reward flat. That is what you would expect -- reward only
arrives for ground never reached before, and random steering almost never
reaches any. The autopilot already gets to about 84 of the 90 checkpoints,
so it can show the policy what the good part of the space looks like instead
of leaving it to find it by accident.

This drives the SAME FZeroEnv the agent trains in, so the observations
recorded are exactly the ones the policy will see, and the episodes end for
exactly the same reasons. Only the choice of action differs.

    python tools/launch_instances.py 8
    python training/record_teacher.py --instances 8 --episodes 40

Writes to the run store (training/run_store.py); build a policy from it with
training/behaviour_cloning.py.

The teacher is stateful and the student is not, quite: the controller
carries a smoothed steering demand, a duty-cycle accumulator and a latched
recovery direction, none of which are in the observation. A memoryless
policy cannot reproduce its duty-cycled steering exactly, and will land on
something closer to the average action. Frame stacking gives the student
some of that history back, and PPO refines the rest -- but it is the reason
to expect a good starting point rather than an autopilot clone.
"""

import argparse
import json
import os
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bridge_client import BridgeTimeout
from control import SCAN_FRACTIONS, Controller
from env.fzero_env import DEFAULT_FRAME_SKIP, FZeroEnv
from training.run_store import RecordingWrapper, RunWriter, new_run_id
from vision.road_edges import measure_rows

DEFAULT_EPISODES = 20


def teacher_action(controller, frame, speed):
    """What the autopilot would do with this frame, as (steer, shoulder).

    Controller.update() still returns (steer, throttle, shoulder) -- it was
    written when the game genuinely had a throttle choice. FZeroEnv now holds
    the throttle down itself and its action_space is MultiDiscrete([3, 3]),
    steer and shoulder only, so the middle value is dropped here rather than
    passed through: sending the 3-tuple straight to env.step() raises
    "too many values to unpack" on the first frame.

    Falls back to holding the last decision when the frame cannot be read;
    the controller already keeps a last signal for exactly that.
    """
    if frame is None:
        steer, _throttle, shoulder = controller.update([], 128.0, speed)[0]
        return steer, shoulder

    height, width = frame.shape[:2]
    readings = measure_rows(frame, [int(height * f) for f in SCAN_FRACTIONS])
    steer, _throttle, shoulder = controller.update(readings, width / 2.0, speed)[0]
    return steer, shoulder


def load_controller_settings(path):
    """Controller keyword arguments from a genome written by genetic.py.

    Returns {} when no file is given, which is the hand-tuned controller.
    """
    if not path:
        return {}

    with open(path) as fh:
        payload = json.load(fh)

    genome = payload.get("genome", payload)
    return {
        "deadzone": genome["deadzone"],
        "full_scale": genome["full_scale"],
        "lean_threshold": genome["lean_threshold"],
        "curve_slowdown": genome["curve_slowdown"],
        "min_corner_speed": genome["min_corner_speed"],
        "smoothing": genome["smoothing"],
        "reversal_hold": int(genome["reversal_hold"]),
        "row_weights": (genome["row_near"], genome["row_mid"], genome["row_far"]),
        "target_clearance": genome["target_clearance"],
    }


def drive(instance, run_id, episodes, frame_skip, results, settings=None):
    environment = FZeroEnv(instance=instance, frame_skip=frame_skip)
    writer = RunWriter(
        run_id,
        instance=instance,
        meta={
            "source": "autopilot",
            "frame_skip": frame_skip,
            "note": "behaviour-cloning teacher data",
            "controller": "evolved" if settings else "hand-tuned",
        },
    )
    recorder = RecordingWrapper(environment, writer)

    done = 0
    try:
        for _ in range(episodes):
            observation, info = recorder.reset()
            # One controller per episode: its smoothing and latched recovery
            # direction are about the run in progress, and carrying them
            # across a reset starts the next episode mid-correction.
            controller = Controller(**(settings or {}))
            speed = info.get("speed", 0)

            while True:
                action = teacher_action(controller, environment.last_frame, speed)
                observation, reward, terminated, truncated, info = recorder.step(action)
                speed = info.get("speed", 0)
                if terminated or truncated:
                    break

            done += 1
            results.append(
                {
                    "instance": instance,
                    "furthest": info.get("furthest", 0),
                    "finished": info.get("finished", False),
                }
            )
    except BridgeTimeout as exc:
        print(f"\n  instance {instance}: bridge stopped responding -- {exc}")
    except Exception as exc:  # noqa: BLE001 - one bad emulator must not kill the rest
        print(f"\n  instance {instance}: {type(exc).__name__}: {exc}")
    finally:
        recorder.close()

    print(f"  instance {instance}: {done} episodes recorded")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0,
                        help="first instance number to use -- match "
                             "launch_instances.py's --offset when running "
                             "alongside emulators already up elsewhere")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="episodes PER emulator")
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--controller", default=None,
                        help="genome JSON from training/genetic.py "
                             "(default: the hand-tuned controller)")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    from training.session_lock import EmulatorSessionLock

    with EmulatorSessionLock(
        "autopilot teacher recording", offset=args.offset, count=max(args.instances, 1)
    ):
        _run(args)


def _run(args):
    settings = load_controller_settings(args.controller)
    if settings:
        print(f"Using the evolved controller from {args.controller}")

    run_id = args.run_id or new_run_id("teacher")
    print(f"Recording {args.episodes} episodes on each of {args.instances} "
          f"emulators into run '{run_id}'.\n")

    started = time.time()
    results = []
    threads = []

    for instance in range(args.offset, args.offset + args.instances):
        thread = threading.Thread(
            target=drive,
            args=(instance, run_id, args.episodes, args.frame_skip, results,
                  settings),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    elapsed = time.time() - started
    print(f"\n{len(results)} episodes in {elapsed / 60:.1f} minutes")

    if results:
        furthest = [r["furthest"] for r in results]
        laps = sum(1 for r in results if r["finished"])
        furthest.sort()
        print(f"furthest checkpoint: median {furthest[len(furthest) // 2]}, "
              f"best {furthest[-1]}")
        print(f"completed laps     : {laps}")

    print(f"\nrun id: {run_id}")
    print(f"Build a policy from it:\n"
          f"    .venv\\Scripts\\python.exe training\\behaviour_cloning.py --run {run_id}")


if __name__ == "__main__":
    main()
