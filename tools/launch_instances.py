"""Start several BizHawk instances for parallel training.

Each emulator gets its own action/state files and its own shared-memory
name, keyed off FZERO_INSTANCE in its environment (lua/bridge.lua reads it).

    python tools/launch_instances.py 4        # start 4
    python tools/launch_instances.py --stop   # stop all of them

Note the instances share BizHawk's config file, so they can overwrite each
other's settings on exit. That does not affect training, but do not expect
GUI changes made in one to stick.
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bridge_client import BASE_DIR, instance_paths

BIZHAWK_DIR = r"C:\Users\chipw\OneDrive\Desktop\BizHawk-2.11.1-win-x64"
EMUHAWK = os.path.join(BIZHAWK_DIR, "EmuHawk.exe")
ROM = os.path.join(BIZHAWK_DIR, "F-Zero (USA).sfc")
LUA_SCRIPT = os.path.join(BASE_DIR, "lua", "bridge.lua")

STARTUP_TIMEOUT = 90.0

# Fixed-frame steering replays from the same savestate showed that 25% and
# 50% have equivalent steering authority, while 100% was about 11% weaker on
# average and 2.6x more variable. Force the fastest measured stable mode for
# every training emulator instead of inheriting a potentially unsafe value.
TRAINING_SPEED_MODE = 50

# Slot every episode resets to. Slot 2 is the purpose-built start state
# from tools/make_start_state.py -- centred, at speed, full power -- which
# gives the agent the same well-conditioned position every episode instead
# of a standing start it barely revisits. Set to None for the hand-saved
# start line in slot 1.
START_SLOT = int(os.environ.get("FZERO_START_SLOT", "2"))


def training_environment(instance, model_label=None):
    """Environment shared by every normal training emulator."""
    env = dict(os.environ)
    env["FZERO_INSTANCE"] = str(instance)
    env["FZERO_SPEEDMODE"] = str(TRAINING_SPEED_MODE)
    if model_label:
        env["FZERO_LABEL"] = str(model_label)
    if START_SLOT is not None:
        env["FZERO_START_SLOT"] = str(START_SLOT)
    return env


def stop_all():
    subprocess.run(
        ["taskkill", "/F", "/IM", "EmuHawk.exe"],
        capture_output=True,
        text=True,
    )
    print("Stopped all EmuHawk instances.")


def state_is_fresh(path, started_at):
    try:
        return os.path.getmtime(path) >= started_at
    except OSError:
        return False


def launch(count, offset=0, model_label=None):
    """Start `count` emulators at instance numbers offset..offset+count-1.

    offset exists so a second, independent batch of emulators can run
    alongside one already up without colliding on instance files or
    shared-memory names -- e.g. gathering more data while another training
    run occupies instances 0-7. stop_all() below kills every EmuHawk by
    name, which would take out that other batch too, so an offset launch
    must be stopped by PID (this function's return value), never by --stop.
    """
    for path in (EMUHAWK, ROM, LUA_SCRIPT):
        if not os.path.exists(path):
            raise SystemExit(f"missing: {path}")

    os.makedirs(os.path.join(BASE_DIR, "frames"), exist_ok=True)

    started_at = time.time()
    processes = []

    for i in range(offset, offset + count):
        action_file, state_file, _, mmf_name = instance_paths(i)

        # Clear stale endpoints so "is it up yet" cannot be answered by a
        # leftover file from a previous run.
        for path in (action_file, state_file):
            try:
                os.remove(path)
            except OSError:
                pass

        env = training_environment(i, model_label=model_label)
        processes.append(
            subprocess.Popen(
                [EMUHAWK, ROM, f"--lua={LUA_SCRIPT}"],
                env=env,
                cwd=BIZHAWK_DIR,
            )
        )
        print(f"  launched instance {i} (pid {processes[-1].pid}, mmf {mmf_name})")

        # Staggered, because several emulators booting at once contend badly
        # on disk and some fail to come up.
        time.sleep(2.0)

    print(f"\nWaiting for {count} bridges to report in...")

    deadline = time.time() + STARTUP_TIMEOUT
    ready = set()
    indices = range(offset, offset + count)

    while time.time() < deadline and len(ready) < count:
        for i in indices:
            if i in ready:
                continue
            _, state_file, _, _ = instance_paths(i)
            if state_is_fresh(state_file, started_at):
                ready.add(i)
                print(f"  instance {i} ready")
        time.sleep(0.5)

    missing = [i for i in indices if i not in ready]
    if missing:
        print(f"\n! instances {missing} never wrote their state file.")
        print("  Check those BizHawk windows for a Lua Console error.")
        return 1, processes

    print(f"\nAll {count} instances ready.")
    return 0, processes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count", type=int, nargs="?", default=4, help="how many emulators")
    parser.add_argument("--offset", type=int, default=0,
                        help="first instance number to use (default 0). Set "
                             "this to run alongside emulators already up on "
                             "the default 0..count-1 range without colliding "
                             "-- but see the warning under --stop.")
    parser.add_argument("--stop", action="store_true",
                        help="kill ALL EmuHawk processes, including ones "
                             "started with --offset by a different run. To "
                             "stop only an offset batch, kill the PIDs this "
                             "script printed when launching it instead.")
    args = parser.parse_args()

    if args.stop:
        stop_all()
        return 0

    code, _processes = launch(args.count, args.offset)
    return code


if __name__ == "__main__":
    sys.exit(main())
