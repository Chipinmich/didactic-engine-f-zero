r"""Capture clean savestates shortly before difficult turns.

This is deliberately separate from training and protected by the emulator
session lock. It never runs automatically and therefore cannot overwrite a
slot while PPO, DQN, or another GA owns the emulators.

Example, using the completed linear seven-ray driver as the scout::

    .venv\Scripts\python.exe tools\make_turn_states.py \
        --model models\neuro_rays_mlp_v11_lineage.npz \
        --targets 28:5,40:6,52:7,64:8,76:9

Each captured state is added to ``models/checkpoint_start_slots.txt``.  The
overnight GA launcher reads that file and evaluates every genome from every
saved location.  It falls back to the two known-safe starts (2 and 4) until
the first state is captured.

Slots are deliberately written only after a safe arrival.  Do not add a slot
by hand: slot 3 is known to start against a wall.
"""

import argparse
import os
import subprocess
import sys
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from env.fzero_env import FZeroEnv
from training.neuroevolution import Stacker, act
from training.session_lock import EmulatorSessionLock
from vision.road_edges import NUM_FEATURES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIZHAWK_DIR = r"C:\Users\chipw\OneDrive\Desktop\BizHawk-2.11.1-win-x64"
EMUHAWK = os.path.join(BIZHAWK_DIR, "EmuHawk.exe")
ROM = os.path.join(BIZHAWK_DIR, "F-Zero (USA).sfc")
LUA = os.path.join(BASE_DIR, "lua", "bridge.lua")
DEFAULT_START_SLOTS_OUT = os.path.join(BASE_DIR, "models", "checkpoint_start_slots.txt")


def parse_targets(value):
    targets = []
    for item in value.split(","):
        checkpoint, slot = (int(part.strip()) for part in item.split(":"))
        if checkpoint < 0 or not 0 <= slot <= 9:
            raise argparse.ArgumentTypeError("targets must look like 28:3,36:4")
        targets.append((checkpoint, slot))
    if len({slot for _, slot in targets}) != len(targets):
        raise argparse.ArgumentTypeError("each target needs a different save slot")
    return tuple(sorted(targets))


def load_saved_slots(path, source_slot):
    """Preserve only known-safe bases and states captured by this tool."""
    slots = [2, 4]
    try:
        with open(path, encoding="utf-8") as handle:
            slots = [int(part.strip()) for part in handle.read().split(",") if part.strip()]
    except OSError:
        pass
    if source_slot not in slots:
        slots.append(source_slot)
    return tuple(dict.fromkeys(slot for slot in slots if 0 <= slot <= 9))


def save_slots(path, slots):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + f".{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(",".join(str(slot) for slot in slots) + "\n")
    os.replace(temporary, path)


def launch(instance, source_slot):
    environment = dict(
        os.environ,
        FZERO_INSTANCE=str(instance),
        FZERO_START_SLOT=str(source_slot),
    )
    return subprocess.Popen(
        [EMUHAWK, ROM, f"--lua={LUA}"],
        env=environment,
        cwd=BIZHAWK_DIR,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.path.join("models", "neuro_rays_mlp_v11_lineage.npz"))
    parser.add_argument(
        "--targets", type=parse_targets,
        default=parse_targets("28:5,40:6,52:7,64:8,76:9"),
        help="evenly distributed checkpoint:slot targets (default: 28:5,40:6,52:7,64:8,76:9)",
    )
    parser.add_argument("--instance", type=int, default=9)
    parser.add_argument(
        "--source-slot",
        type=int,
        default=2,
        help="savestate slot the scout starts from (default: 2)",
    )
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--start-slots-out", default=DEFAULT_START_SLOTS_OUT,
        help="verified checkpoint slot list consumed by the overnight GA launcher",
    )
    args = parser.parse_args()

    with np.load(args.model) as model:
        genome = model["weights"].astype(np.float32)
        hidden = int(model["hidden"])
        inputs = int(model["inputs"])
    if inputs % NUM_FEATURES:
        raise SystemExit(f"model has incompatible observation width: {inputs}")
    stack = inputs // NUM_FEATURES

    if not 0 <= args.source_slot <= 9:
        raise SystemExit("--source-slot must be between 0 and 9")

    process = launch(args.instance, args.source_slot)
    environment = None
    try:
        time.sleep(14)
        environment = FZeroEnv(
            instance=args.instance,
            frame_skip=4,
            assist_mode="full",
            start_slots=(args.source_slot,),
        )
        observation, info = environment.reset()
        stacker = Stacker(stack)
        stacked = stacker.reset(observation)
        pending = list(args.targets)
        saved_slots = list(load_saved_slots(args.start_slots_out, args.source_slot))

        while pending:
            code = act(genome, stacked, inputs, hidden)
            observation, _, terminated, truncated, info = environment.step(
                (code // 3, code % 3)
            )
            stacked = stacker.push(observation)
            checkpoint, slot = pending[0]
            safe = (
                info.get("furthest", 0) >= checkpoint
                and not info.get("wall", False)
                and info.get("speed", 0) >= 500
                and info.get("off_centre", 1.0) <= 0.45
            )
            if safe:
                environment.bridge.save(slot)
                if slot not in saved_slots:
                    saved_slots.append(slot)
                save_slots(args.start_slots_out, saved_slots)
                print(
                    f"saved checkpoint {info.get('furthest')} to slot {slot} "
                    f"(speed {info.get('speed')}, off-centre "
                    f"{info.get('off_centre', 0.0) * 100:.0f}%)"
                )
                pending.pop(0)
            if terminated or truncated:
                missing = ", ".join(f"cp {cp} -> slot {slot}" for cp, slot in pending)
                raise SystemExit(f"scout ended before capturing: {missing}")
    finally:
        if environment is not None:
            environment.close()
        if not args.keep:
            process.terminate()

    print(
        "Checkpoint curriculum updated: --start-slots "
        + ",".join(str(slot) for slot in load_saved_slots(args.start_slots_out, args.source_slot))
    )


if __name__ == "__main__":
    with EmulatorSessionLock("turn-state capture"):
        main()
