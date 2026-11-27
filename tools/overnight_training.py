"""Run GA, DQN, and PPO together, either fresh or as one continuous lineage.

This is intentionally conservative: it refuses to start if another trainer
or an unmanaged EmuHawk window is active. It never uses the project's global
``--stop`` operation; cleanup is limited to the exact emulator processes this
launcher created.
"""

import argparse
import glob
import json
import os
import pickle
import signal
import subprocess
import sys
import time
import zipfile
from datetime import datetime

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_DIR)

from tools.launch_instances import launch  # noqa: E402
from training.session_lock import EmulatorSessionLock  # noqa: E402


# Three learners, four emulators each, on identical wall-clock time. Equal
# allocation is the whole point: it is the one comparison between a population
# search and two gradient methods that needs no normalising, because per-episode
# and per-update counts are not comparable across them by design.
#
# 4+4+4 uses 12 of the 14 rather than splitting 14 three ways as 5/5/4. An
# uneven split would hand one learner a 25% hardware advantage over another and
# there is no principled way to choose which. The two spare instances are also
# genuinely useful: every diagnostic probe in this project so far -- steering
# authority, speed-mode fidelity, RAM capture -- has needed a free emulator,
# and each one previously cost a full training restart to get.
GA_INSTANCES = 4
GA_OFFSET = 0
DQN_INSTANCES = 4
DQN_OFFSET = 4
PPO_INSTANCES = 4
PPO_OFFSET = 8

# Emulator instances this machine runs. Measured ceiling, not a guess: past 8
# the single Python process driving them is the bottleneck, so extra instances
# split the same throughput -- but the three trainers are separate processes,
# so 14 across them is real capacity.
TOTAL_INSTANCES = 14

# 12 and 13 are deliberately left free for diagnostics in the fresh three-way
# split. Inherited runs use all 14; see instance_plan().
SPARE_INSTANCES = (12, 13)

# --inherit runs the two GRADIENT learners only, seven emulators each, using
# all 14. The genetic search is dropped rather than squeezed.
#
# On the measurement that matters -- distance actually travelled, which is
# immune to the progress-accounting bugs the other metrics carried -- PPO ran
# 532,468 per episode against the GA's 251,742 and the DQN's 219,445, so the
# GA was the weakest of the three while holding a third of the hardware. The
# two gradient learners are also the ones with a real comparison left to
# settle, and giving each seven rather than four is close to a doubling of
# their sample rate.
INHERIT_DQN_INSTANCES = 7
INHERIT_DQN_OFFSET = 0
INHERIT_PPO_INSTANCES = 7
INHERIT_PPO_OFFSET = 7


def instance_plan(inherit):
    """(label, screen label, count, offset) per learner, for this mode.

    Fresh runs compare three learners on equal hardware; inherited runs drop
    the genetic search and split all 14 between the two gradient learners.
    """
    if inherit:
        return (
            ("DQN", "DQN", INHERIT_DQN_INSTANCES, INHERIT_DQN_OFFSET),
            ("PPO", "PPO", INHERIT_PPO_INSTANCES, INHERIT_PPO_OFFSET),
        )
    return (
        ("genetic", "NEUROEVOLUTION", GA_INSTANCES, GA_OFFSET),
        ("DQN", "DQN", DQN_INSTANCES, DQN_OFFSET),
        ("PPO", "PPO", PPO_INSTANCES, PPO_OFFSET),
    )


DEFAULT_PPO_RESTARTS = 3
INHERITED_WALL_CONTACT_FRAMES = 20
CHECKPOINT_START_SLOTS_FILE = os.path.join(
    PROJECT_DIR, "models", "checkpoint_start_slots.txt"
)
MODEL_DIR = os.path.join(PROJECT_DIR, "models")
GA_CHECKPOINT = os.path.join(MODEL_DIR, "neuro_rays_mlp_v11_lineage.npz.state.pkl")


def checkpoint_start_slots():
    """Return the verified distributed starts, or the two known-safe starts.

    The checkpoint capture tool writes this file only after it has saved a
    usable BizHawk state.  Never guess that slots 5-9 exist: an empty or bad
    state turns a generalisation curriculum into a menu/crash curriculum.
    """
    fallback = (2, 4)
    try:
        with open(CHECKPOINT_START_SLOTS_FILE, encoding="utf-8") as handle:
            values = tuple(int(part.strip()) for part in handle.read().split(",") if part.strip())
    except (OSError, ValueError):
        return fallback
    if not values or len(set(values)) != len(values) or any(slot < 0 or slot > 9 for slot in values):
        return fallback
    return values


def build_ga_command(python, generations=100, resume=False,
                     wall_contact_frames=8, fork_from=None):
    start_slots = checkpoint_start_slots()
    command = [
        python,
        "-u",
        os.path.join(PROJECT_DIR, "training", "neuroevolution.py"),
        "--instances", str(GA_INSTANCES),
        "--offset", str(GA_OFFSET),
        "--generations", str(generations),
        "--hidden", "32",
        "--population", "48",
        # v11 starts from nothing under the one-checkpoint distance-band
        # ranking and lineage logging. It must not silently seed itself from
        # v10, whose continuous composite made turn and centering terms inert.
        # Every population before v10 was also ranked on RAM
        # 0x000C1, which is the lap timer, not track position -- so roughly
        # 40,000 episodes of selection pressure rewarded elapsed time. Those
        # genomes are not a warm start, they are noise with a pedigree.
        "--out", os.path.join(PROJECT_DIR, "models", "neuro_rays_mlp_v11_lineage.npz"),
        # NO --fork. v4 -> v5 -> v6 all descend from a straight-line lineage:
        # v4's elite carried output bias +1.125 on "steer straight" and chose
        # straight on 98.5% of 1415 real observations, and v5's population
        # still leaned that way (elite bias gap +0.485, 39.6% of genomes
        # picking straight on a blank input). Every fork carried it forward.
        #
        # NO teacher seed either, and --no-teacher-seed alone was NOT enough:
        # it skipped the recorded-autopilot seed but fell through to
        # analytic_teacher_seed, which is the hand-written controller -- and
        # that controller drives straight until it meets a wall. It seeded up
        # to six genomes with precisely the behaviour being escaped.
        "--no-teacher-seed",
        # The correction seed STAYS, on measurement. It is not
        # autopilot-derived: it is built from safety_override frames in a
        # prior GA run, mirror-augmented, and measures 0% straight /
        # 50% left / 50% right. It is the one component actively teaching
        # wall-avoidance turning, so removing it would delete the cure and
        # keep the disease.
        "--correction-run", "auto",
        "--evaluation-seconds", "30",
        "--end-on-wall",
        # Eight real consecutive contact frames gives the policy a brief
        # chance to correct a scrape, while still ending sustained wall
        # grinding early. This is deliberately a termination rule, not a
        # fitness component, so distance remains the primary objective.
        "--wall-contact-frames", str(wall_contact_frames),
        "--final-repeats", "12",
        # Slot 3 is DELIBERATELY EXCLUDED -- do not re-add it without
        # rebuilding that savestate first. Measured over 8700 episodes
        # (run neuro_20260821_030117, 2900 episodes per slot):
        #
        #   slot  start_cp  steps  progress_gain  wall_contact_rate
        #     2      17.1    74.0      13.95           6.0%
        #     3      32.0    16.5       1.93          31.7%
        #     4      20.0    56.4      13.95           7.8%
        #
        # Slot 3 starts the machine essentially against a wall: a third of
        # its frames are in contact from the opening step, every episode
        # ends inside ~16 steps, and it gains ~2 checkpoints no matter which
        # genome drives (stdev 1.16). That is a third of the evaluation
        # budget spent replaying a near-identical crash, and it adds a
        # near-constant to every genome's mean, diluting the fitness
        # differences the search needs to see.
        "--start-slots", ",".join(str(slot) for slot in start_slots),
    ]
    # Each candidate must face every saved position in the same generation;
    # otherwise selection can mistake an easy checkpoint for a good genome.
    if len(start_slots) > 6:
        command.extend(["--repeats", str(len(start_slots))])
    if resume:
        command.extend(["--resume", "auto"])
    if fork_from:
        command.extend(["--fork", fork_from])
    return command


def build_dqn_command(python, timesteps=1_500_000, run_id=None, resume=False,
                     instances=DQN_INSTANCES, offset=DQN_OFFSET,
                     wall_contact_frames=None):
    if run_id is None:
        run_id = "dqn_overnight_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    command = [
        python,
        "-u",
        os.path.join(PROJECT_DIR, "training", "train_dqn.py"),
        "--instances", str(instances),
        "--offset", str(offset),
        "--timesteps", str(timesteps),
        # Fresh, and NOT --resume, for the same reason the GA is fresh: the
        # env reward is built on progress, so the 1.6M steps in dqn_proper
        # were shaped by the lap timer too. A unique name keeps both the model
        # and dashboard episodes separate from every previous overnight run.
        "--name", run_id,
        "--assist-mode", "emergency",
        "--run-id", run_id,
        # The SAME spawns the genetic search uses. Training from one position
        # teaches that position's corner sequence, and a policy that memorised
        # a layout cannot be told apart from one that learned to drive -- so
        # varying the start is both a generalisation measure and the only way
        # the comparison stays fair, since all three must face the same task.
        "--start-slots", ",".join(str(slot) for slot in checkpoint_start_slots()),
    ]
    if resume:
        command.append("--resume")
    if wall_contact_frames is not None:
        command.extend(["--wall-contact-frames", str(wall_contact_frames)])
    return command


def build_ppo_command(python, timesteps=1_500_000, run_id=None, resume=False,
                     instances=PPO_INSTANCES, offset=PPO_OFFSET,
                     wall_contact_frames=None):
    """PPO, in the same environment as the other two.

    --assist-mode had to be added to train.py for this: it was hard-wired to
    "full" while DQN runs "emergency", so a head-to-head would have compared
    environments as much as algorithms.
    """
    if run_id is None:
        run_id = "ppo_overnight_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    command = [
        python,
        "-u",
        os.path.join(PROJECT_DIR, "training", "train.py"),
        "--instances", str(instances),
        "--offset", str(offset),
        "--timesteps", str(timesteps),
        # Fresh, like the other two. Every PPO run before now was trained
        # against a reward whose progress term was the lap timer, and then
        # against one that paid the agent to end the episode early.
        "--name", run_id,
        "--run-id", run_id,
        "--assist-mode", "emergency",
        # The SAME spawns the genetic search uses. Training from one position
        # teaches that position's corner sequence, and a policy that memorised
        # a layout cannot be told apart from one that learned to drive -- so
        # varying the start is both a generalisation measure and the only way
        # the comparison stays fair, since all three must face the same task.
        "--start-slots", ",".join(str(slot) for slot in checkpoint_start_slots()),
    ]
    if resume:
        command.append("--resume")
    if wall_contact_frames is not None:
        command.extend(["--wall-contact-frames", str(wall_contact_frames)])
    return command


def ga_checkpoint_summary(path=GA_CHECKPOINT):
    try:
        with open(path, "rb") as handle:
            state = pickle.load(handle)
        return {
            "generations": max(int(state.get("next_generation", 0)), 0),
            "wall_contact_frames": int(
                state.get("config", {}).get("wall_contact_frames", 0)
            ),
        }
    except (OSError, EOFError, TypeError, ValueError, pickle.UnpicklingError) as exc:
        raise SystemExit(f"Cannot inherit GA checkpoint {path}: {exc}") from exc


def ga_completed_generations(path=GA_CHECKPOINT):
    return ga_checkpoint_summary(path)["generations"]


def latest_continuous_run(prefix, required_suffix):
    """Find the newest final model that has its algorithm-specific state."""
    candidates = []
    for path in glob.glob(os.path.join(MODEL_DIR, f"{prefix}_*.zip")):
        name = os.path.basename(path)
        if name[:-4].endswith("_steps") or "_steps_" in name:
            continue
        run_id = name[:-4]
        required = os.path.join(MODEL_DIR, run_id + required_suffix)
        if os.path.exists(required):
            candidates.append((os.path.getmtime(path), run_id))
    if not candidates:
        raise SystemExit(
            f"No inheritable {prefix} model with {required_suffix} was found in {MODEL_DIR}"
        )
    return max(candidates)[1]


def saved_model_timesteps(model_path):
    """Read SB3's saved counter without constructing a policy or emulator."""
    try:
        with zipfile.ZipFile(model_path + ".zip") as archive:
            payload = json.loads(archive.read("data").decode("utf-8"))
        return max(int(payload.get("num_timesteps", 0)), 0)
    except (OSError, KeyError, ValueError, TypeError, zipfile.BadZipFile, json.JSONDecodeError):
        return None


def build_ppo_recovery_command(python, target_timesteps, run_id,
                               wall_contact_frames=None):
    """Resume only the work PPO had not saved before an emulator failure."""
    model_path = os.path.join(PROJECT_DIR, "models", run_id)
    completed = saved_model_timesteps(model_path)
    if completed is not None and completed >= target_timesteps:
        return None, completed
    remaining = target_timesteps if completed is None else target_timesteps - completed
    command = build_ppo_command(
        python, remaining, run_id=run_id,
        wall_contact_frames=wall_contact_frames,
    )
    if completed is not None:
        command.append("--resume")
    return command, completed or 0


def emuhawk_is_running():
    result = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq EmuHawk.exe"],
        capture_output=True,
        text=True,
        check=False,
    )
    return "emuhawk.exe" in result.stdout.lower()


def require_idle_machine():
    # A whole-space preflight catches every live trainer range and also
    # sweeps stale lock records. It is released before child trainers start.
    with EmulatorSessionLock("overnight launcher preflight"):
        pass
    if emuhawk_is_running():
        raise SystemExit(
            "EmuHawk is already running without a trainer lock. Close those "
            "emulator windows before starting the overnight run. Nothing "
            "was stopped or changed."
        )


def stop_exact_processes(processes, grace=8.0):
    alive = [process for process in processes if process.poll() is None]
    for process in alive:
        process.terminate()
    deadline = time.time() + grace
    while alive and time.time() < deadline:
        alive = [process for process in alive if process.poll() is None]
        time.sleep(0.2)
    for process in alive:
        process.kill()


def interrupt_trainers(trainers):
    for process in trainers:
        if process.poll() is not None:
            continue
        try:
            if os.name == "nt":
                os.kill(process.pid, signal.CTRL_BREAK_EVENT)
            else:
                process.send_signal(signal.SIGINT)
        except OSError:
            pass

    deadline = time.time() + 30.0
    while time.time() < deadline:
        if all(process.poll() is not None for process in trainers):
            return
        time.sleep(0.25)
    stop_exact_processes(trainers)


def write_status(path, phase, ga=None, dqn=None, emulators=(), ppo=None,
                 ppo_restarts=0):
    payload = {
        "phase": phase,
        "updated": time.time(),
        "ga_pid": None if ga is None else ga.pid,
        "dqn_pid": None if dqn is None else dqn.pid,
        "ppo_pid": None if ppo is None else ppo.pid,
        "ppo_restarts": int(ppo_restarts),
        "emulator_pids": [process.pid for process in emulators],
    }
    temporary = path + f".{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ga-generations", type=int, default=100)
    parser.add_argument("--dqn-timesteps", type=int, default=1_500_000)
    parser.add_argument("--ppo-timesteps", type=int, default=1_500_000)
    parser.add_argument(
        "--ppo-restarts", type=int, default=DEFAULT_PPO_RESTARTS,
        help="times to relaunch PPO after an emulator/bridge failure (default: 3)",
    )
    parser.add_argument(
        "--inherit", action="store_true",
        help="continue the latest GA population, DQN replay lineage, and PPO lineage",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="show the two commands without checking or starting anything",
    )
    args = parser.parse_args()
    if args.ga_generations < 1 or args.dqn_timesteps < 1 or args.ppo_timesteps < 1:
        parser.error("generation and timestep counts must be positive")
    if args.ppo_restarts < 0:
        parser.error("--ppo-restarts cannot be negative")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.inherit:
        # No genetic search when inheriting: the two gradient learners take
        # seven emulators each. See instance_plan() for why. The GA checkpoint
        # is untouched, so a fresh three-way run still picks it up.
        ga_command = None
        ga_inheritance = "not running (gradient learners take all 14 emulators)"
        dqn_run_id = latest_continuous_run("dqn_overnight", "_replay_buffer.pkl")
        ppo_run_id = latest_continuous_run("ppo_overnight", "_vecnorm.pkl")
        ppo_saved_steps = saved_model_timesteps(os.path.join(MODEL_DIR, ppo_run_id))
        if ppo_saved_steps is None:
            raise SystemExit(f"Cannot read inherited PPO step count for {ppo_run_id}")
        ppo_target_total = ppo_saved_steps + args.ppo_timesteps
        dqn_command = build_dqn_command(
            sys.executable, args.dqn_timesteps, run_id=dqn_run_id, resume=True,
            instances=INHERIT_DQN_INSTANCES, offset=INHERIT_DQN_OFFSET,
            wall_contact_frames=INHERITED_WALL_CONTACT_FRAMES,
        )
        ppo_command = build_ppo_command(
            sys.executable, args.ppo_timesteps, run_id=ppo_run_id, resume=True,
            instances=INHERIT_PPO_INSTANCES, offset=INHERIT_PPO_OFFSET,
            wall_contact_frames=INHERITED_WALL_CONTACT_FRAMES,
        )
        print(
            f"Inherited lineage: GA {ga_inheritance}; "
            f"DQN {dqn_run_id}; PPO {ppo_run_id} at {ppo_saved_steps} steps."
        )
    else:
        dqn_run_id = f"dqn_overnight_{stamp}"
        ppo_run_id = f"ppo_overnight_{stamp}"
        ppo_target_total = args.ppo_timesteps
        ga_command = build_ga_command(sys.executable, args.ga_generations)
        dqn_command = build_dqn_command(
            sys.executable, args.dqn_timesteps, run_id=dqn_run_id
        )
        ppo_command = build_ppo_command(
            sys.executable, args.ppo_timesteps, run_id=ppo_run_id
        )
    plan = instance_plan(args.inherit)
    if ga_command is not None:
        print("Genetic:  " + subprocess.list2cmdline(ga_command))
    print("DQN:      " + subprocess.list2cmdline(dqn_command))
    print("PPO:      " + subprocess.list2cmdline(ppo_command))
    # Derived, not written out by hand: it said "genetic 0-7, DQN 8-13" for a
    # while after the split changed, which is exactly the kind of quietly
    # wrong status line that costs an hour later.
    used = {i for _, _, count, offset in plan for i in range(offset, offset + count)}
    ranges = ", ".join(
        f"{label} {offset}-{offset + count - 1}" for label, _, count, offset in plan
    )
    idle = sorted(set(range(TOTAL_INSTANCES)) - used)
    print(
        f"Instance ranges: {ranges} (no overlap)"
        + (f"; {','.join(str(i) for i in idle)} left free for diagnostics." if idle else ".")
    )
    if args.dry_run:
        print("Dry run only: no emulators or trainers were started.")
        return 0

    require_idle_machine()

    log_dir = os.path.join(PROJECT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    status_path = os.path.join(log_dir, f"overnight_{stamp}_status.json")
    logs = {}
    emulators = []
    ppo_emulators = []
    trainers = []
    ga = dqn = ppo = None
    ppo_restart_count = 0
    ppo_target_met = False

    try:
        for label, screen_label, count, offset in plan:
            print(f"Starting {label} emulators {offset}-{offset + count - 1}...")
            code, started = launch(count, offset, model_label=screen_label)
            emulators.extend(started)
            if label == "PPO":
                ppo_emulators[:] = started
            if code:
                raise RuntimeError(f"one or more {label} emulator bridges failed to start")

        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        for label in ("ga", "dqn", "ppo"):
            for stream in ("out", "err"):
                path = os.path.join(log_dir, f"overnight_{stamp}_{label}.{stream}.log")
                logs[f"{label}_{stream}"] = open(path, "w", encoding="utf-8", buffering=1)

        if ga_command is not None:
            ga = subprocess.Popen(
                ga_command, cwd=PROJECT_DIR,
                stdout=logs["ga_out"], stderr=logs["ga_err"],
                creationflags=creation_flags,
            )
        dqn = subprocess.Popen(
            dqn_command, cwd=PROJECT_DIR,
            stdout=logs["dqn_out"], stderr=logs["dqn_err"],
            creationflags=creation_flags,
        )
        ppo = subprocess.Popen(
            ppo_command, cwd=PROJECT_DIR,
            stdout=logs["ppo_out"], stderr=logs["ppo_err"],
            creationflags=creation_flags,
        )
        trainers[:] = [t for t in (ga, dqn, ppo) if t is not None]
        write_status(status_path, "training", ga, dqn, emulators, ppo,
                     ppo_restart_count)
        running = ", ".join(
            f"{name} {process.pid}"
            for name, process in (("genetic", ga), ("DQN", dqn), ("PPO", ppo))
            if process is not None
        )
        print(f"{len(trainers)} trainers are running ({running}).")
        print(f"Logs and status: {log_dir}")
        print(f"Dashboard run IDs: DQN {dqn_run_id}, PPO {ppo_run_id}.")
        print(r"Compare them: .venv\Scripts\python.exe tools\compare_learners.py")

        while True:
            if (
                not ppo_target_met
                and
                ppo.poll() is not None
                and ppo.returncode != 0
                and ppo_restart_count < args.ppo_restarts
            ):
                recovery_command, completed = build_ppo_recovery_command(
                    sys.executable, ppo_target_total, ppo_run_id,
                    wall_contact_frames=(
                        INHERITED_WALL_CONTACT_FRAMES if args.inherit else None
                    )
                )
                if recovery_command is None:
                    print(
                        f"PPO saved {completed} steps, meeting its target; "
                        "not restarting despite the cleanup error."
                    )
                    ppo_target_met = True
                    time.sleep(2.0)
                    continue

                ppo_restart_count += 1
                print(
                    f"PPO exited {ppo.returncode} after saving {completed} steps. "
                    f"Restarting its emulators and resuming "
                    f"({ppo_restart_count}/{args.ppo_restarts})."
                )
                stop_exact_processes(ppo_emulators)
                for process in ppo_emulators:
                    if process in emulators:
                        emulators.remove(process)
                code, started = launch(
                    PPO_INSTANCES, PPO_OFFSET, model_label="PPO"
                )
                if code:
                    stop_exact_processes(started)
                    raise RuntimeError("PPO emulator bridges failed during recovery")
                ppo_emulators[:] = started
                emulators.extend(started)
                ppo = subprocess.Popen(
                    recovery_command, cwd=PROJECT_DIR,
                    stdout=logs["ppo_out"], stderr=logs["ppo_err"],
                    creationflags=creation_flags,
                )
                trainers[2] = ppo
                write_status(status_path, "training", ga, dqn, emulators, ppo,
                             ppo_restart_count)
            if not any(process.poll() is None for process in trainers):
                break
            time.sleep(2.0)

        success = (
            ga.returncode == 0
            and dqn.returncode == 0
            and (ppo.returncode == 0 or ppo_target_met)
        )
        write_status(status_path, "complete" if success else "failed", ga, dqn,
                     emulators, ppo, ppo_restart_count)
        print(f"Genetic exited {ga.returncode}; DQN exited {dqn.returncode}; "
              f"PPO exited {ppo.returncode}.")
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\nStopping safely; asking both trainers to save first...")
        interrupt_trainers(trainers)
        write_status(status_path, "interrupted", ga, dqn, emulators, ppo,
                     ppo_restart_count)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"[error] Overnight launch failed: {exc}", file=sys.stderr)
        interrupt_trainers(trainers)
        if os.path.isdir(log_dir):
            write_status(status_path, "failed", ga, dqn, emulators)
        return 1
    finally:
        stop_exact_processes(emulators)
        for handle in logs.values():
            handle.close()


if __name__ == "__main__":
    sys.exit(main())
