"""Offline checks for the combined overnight launcher."""

import os
import sys
import tempfile
import json
import pickle
import zipfile

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from tools.overnight_training import (  # noqa: E402
    TOTAL_INSTANCES,
    instance_plan,
    build_dqn_command,
    build_ga_command,
    build_ppo_command,
    build_ppo_recovery_command,
    ga_completed_generations,
    ga_checkpoint_summary,
    saved_model_timesteps,
    stop_exact_processes,
)
import tools.overnight_training as overnight  # noqa: E402
from tools.launch_instances import TRAINING_SPEED_MODE, training_environment  # noqa: E402


def option(command, name):
    return command[command.index(name) + 1]


def test_training_emulators_force_measured_stable_speed_mode():
    environment = training_environment(7, model_label="DQN")

    assert TRAINING_SPEED_MODE == 50
    assert environment["FZERO_INSTANCE"] == "7"
    assert environment["FZERO_SPEEDMODE"] == "50"
    assert environment["FZERO_LABEL"] == "DQN"


def test_all_three_learners_get_equal_disjoint_instance_ranges():
    """Equal allocation is the comparison; unequal allocation is a confound.

    Per-episode and per-update counts are not comparable between a population
    search and two gradient methods, so identical emulator counts over
    identical wall-clock time is the one basis that needs no normalising.
    14 does not divide by three, and a 5/5/4 split would hand one learner 25%
    more hardware than another with no principled way to pick which, so 12 are
    used and 2 held back for diagnostics.
    """
    ranges = {}
    for name, command in (
        ("ga", build_ga_command("python", 60)),
        ("dqn", build_dqn_command("python", 500_000)),
        ("ppo", build_ppo_command("python", 500_000)),
    ):
        start = int(option(command, "--offset"))
        count = int(option(command, "--instances"))
        ranges[name] = set(range(start, start + count))

    assert len({len(r) for r in ranges.values()}) == 1, (
        f"learners got different emulator counts: "
        f"{ {k: len(v) for k, v in ranges.items()} }"
    )
    assert ranges["ga"] == set(range(0, 4))
    assert ranges["dqn"] == set(range(4, 8))
    assert ranges["ppo"] == set(range(8, 12))
    for a in ranges:
        for b in ranges:
            if a < b:
                assert ranges[a].isdisjoint(ranges[b]), f"{a} and {b} overlap"


def test_ppo_runs_in_the_same_environment_as_the_dqn():
    """Otherwise the comparison measures environments, not algorithms.

    train.py hard-wired assist_mode="full" with no flag, while the DQN runs
    "emergency" -- so PPO would have faced a different amount of safety
    intervention than the learner it is being compared against.
    """
    dqn = build_dqn_command("python", 500_000)
    ppo = build_ppo_command("python", 500_000)
    assert option(ppo, "--assist-mode") == option(dqn, "--assist-mode")
    assert option(ppo, "--timesteps") == option(dqn, "--timesteps")
    assert "--resume" not in ppo, "PPO must not inherit lap-timer-era weights"


def test_ppo_recovery_resumes_only_unsaved_target_steps():
    run_id = "ppo_recovery_test"
    model_path = os.path.join(overnight.PROJECT_DIR, "models", run_id)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    try:
        with zipfile.ZipFile(model_path + ".zip", "w") as archive:
            archive.writestr("data", json.dumps({"num_timesteps": 60_416}))
        assert saved_model_timesteps(model_path) == 60_416
        command, completed = build_ppo_recovery_command(
            "python", 1_500_000, run_id
        )
        assert completed == 60_416
        assert option(command, "--timesteps") == str(1_500_000 - 60_416)
        assert option(command, "--run-id") == run_id
        assert "--resume" in command
    finally:
        os.unlink(model_path + ".zip")


def test_inherited_commands_continue_the_same_lineages():
    ga = build_ga_command("python", 200, resume=True, wall_contact_frames=20)
    dqn = build_dqn_command(
        "python", 1_500_000, run_id="dqn_overnight_parent", resume=True,
        wall_contact_frames=20,
    )
    ppo = build_ppo_command(
        "python", 1_500_000, run_id="ppo_overnight_parent", resume=True,
        wall_contact_frames=20,
    )
    assert option(ga, "--generations") == "200"
    assert option(ga, "--resume") == "auto"
    assert option(dqn, "--run-id") == "dqn_overnight_parent"
    assert option(ppo, "--run-id") == "ppo_overnight_parent"
    assert "--resume" in dqn
    assert "--resume" in ppo
    assert option(ga, "--wall-contact-frames") == "20"
    assert option(dqn, "--wall-contact-frames") == "20"
    assert option(ppo, "--wall-contact-frames") == "20"


def test_inherited_ga_adds_generations_to_checkpoint_count():
    descriptor, path = tempfile.mkstemp(prefix="fzero_ga_state_")
    os.close(descriptor)
    try:
        with open(path, "wb") as handle:
            pickle.dump({
                "next_generation": 100,
                "config": {"wall_contact_frames": 8},
            }, handle)
        assert ga_completed_generations(path) == 100
        assert ga_checkpoint_summary(path)["wall_contact_frames"] == 8
    finally:
        os.unlink(path)


def test_all_three_learners_spawn_from_the_same_varied_positions():
    """Same task for all three, and more than one starting point each.

    Training from a single position teaches that position's corner sequence,
    and a policy that memorised a layout is indistinguishable from one that
    learned to drive. It is also the only way the comparison stays honest:
    if one learner sees four starts and another sees one, they are not being
    asked the same question.
    """
    slots = {
        name: option(command, "--start-slots")
        for name, command in (
            ("ga", build_ga_command("python", 60)),
            ("dqn", build_dqn_command("python", 500_000)),
            ("ppo", build_ppo_command("python", 500_000)),
        )
    }
    assert len(set(slots.values())) == 1, f"learners got different spawns: {slots}"
    assert len(slots["ga"].split(",")) > 1, "a single spawn teaches one layout"


def test_overnight_builder_defaults_match_the_long_run():
    ga = build_ga_command("python")
    dqn = build_dqn_command("python")
    assert option(ga, "--generations") == "100"
    assert option(dqn, "--timesteps") == "1500000"


def test_ga_starts_a_clean_v11_lineage_on_banded_real_progress():
    """v11 must not inherit continuous-score or lap-timer selection.

    Every population before v10 was ranked on RAM 0x000C1, which is the lap
    TIMER rather than track position, so ~40,000 episodes of selection
    pressure rewarded elapsed time. Resuming or forking that lineage would
    carry the result of a broken fitness function into a fixed one, so both
    are excluded here rather than left to a comment.
    """
    original_slots_file = overnight.CHECKPOINT_START_SLOTS_FILE
    overnight.CHECKPOINT_START_SLOTS_FILE = os.path.join(tempfile.gettempdir(), "no-fzero-slots-here.txt")
    try:
        command = build_ga_command("python", 77)
    finally:
        overnight.CHECKPOINT_START_SLOTS_FILE = original_slots_file
    assert option(command, "--generations") == "77"
    assert option(command, "--hidden") == "32"
    assert option(command, "--population") == "48"
    assert option(command, "--start-slots") == "2,4"
    assert option(command, "--evaluation-seconds") == "30"
    # 3, not 45: contact ends the run so wall-grinding is structurally
    # impossible rather than taxed. At 45 the tax had become the objective --
    # 69% contact meant WALL_RATE_PENALTY was worth -28 checkpoints against a
    # mean distance of 26, so selection ranked "touched walls least".
    assert option(command, "--wall-contact-frames") == "8"
    assert "--end-on-wall" in command
    assert "--elite-random-split" not in command
    assert "--no-teacher-seed" in command
    assert "--resume" not in command, "v11 must start with clean ranking evidence"
    assert "--fork" not in command, "v11 must not fork incompatible selection"
    assert option(command, "--out").endswith("neuro_rays_mlp_v11_lineage.npz")


def test_ga_uses_every_verified_checkpoint_start_in_one_generation():
    handle, path = tempfile.mkstemp(prefix="fzero_slots_", text=True)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as slots:
            slots.write("2,4,5,6,7,8,9\n")
        original_slots_file = overnight.CHECKPOINT_START_SLOTS_FILE
        overnight.CHECKPOINT_START_SLOTS_FILE = path
        try:
            command = build_ga_command("python", 30)
        finally:
            overnight.CHECKPOINT_START_SLOTS_FILE = original_slots_file
        assert option(command, "--start-slots") == "2,4,5,6,7,8,9"
        assert option(command, "--repeats") == "7"
    finally:
        os.unlink(path)


def test_dqn_also_starts_clean_rather_than_resuming():
    """The DQN's reward is built on progress, so its history is affected too.

    dqn_proper reached 1.6M steps, all of them shaped by a reward whose
    progress term was the lap timer. It starts fresh under a new name, which
    also leaves the old checkpoint intact instead of overwriting it.
    """
    command = build_dqn_command(
        "python", 123_456, run_id="dqn_overnight_20990102_030405"
    )
    assert option(command, "--timesteps") == "123456"
    assert option(command, "--name") == "dqn_overnight_20990102_030405"
    assert option(command, "--run-id") == "dqn_overnight_20990102_030405"
    assert option(command, "--assist-mode") == "emergency"
    assert "--resume" not in command


class FakeProcess:
    def __init__(self, running):
        self.running = running
        self.terminated = 0
        self.killed = 0

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.terminated += 1
        self.running = False

    def kill(self):
        self.killed += 1
        self.running = False


def test_cleanup_only_touches_supplied_running_processes():
    running = FakeProcess(True)
    finished = FakeProcess(False)
    stop_exact_processes([running, finished], grace=0)
    assert running.terminated == 1
    assert finished.terminated == 0


def test_inherited_runs_give_the_two_gradient_learners_seven_each():
    """No genetic search when inheriting; DQN and PPO take all 14 emulators.

    On distance actually travelled -- the measure immune to the progress
    accounting bugs the others carried -- PPO ran 532,468 per episode against
    the GA's 251,742 and the DQN's 219,445, so the GA was the weakest of the
    three while holding a third of the hardware.
    """
    plan = instance_plan(inherit=True)
    labels = [label for label, _, _, _ in plan]
    assert labels == ["DQN", "PPO"], f"inherited run should drop the GA, got {labels}"

    used = [i for _, _, count, offset in plan for i in range(offset, offset + count)]
    assert len(used) == len(set(used)), "instance ranges overlap"
    assert sorted(used) == list(range(TOTAL_INSTANCES)), (
        "inherited runs should use every emulator"
    )
    assert {count for _, _, count, _ in plan} == {7}, "both learners get seven"


def test_fresh_runs_still_compare_three_learners_equally():
    plan = instance_plan(inherit=False)
    assert [label for label, _, _, _ in plan] == ["genetic", "DQN", "PPO"]
    assert {count for _, _, count, _ in plan} == {4}, (
        "equal allocation is what makes the three-way comparison meaningful"
    )


def test_inherited_commands_match_the_inherited_instance_plan():
    """The commands and the emulators launched must not disagree.

    Launching seven bridges and telling the trainer to use four leaves three
    emulators idle and the trainer waiting on instances nobody started.
    """
    plan = dict((label, (count, offset)) for label, _, count, offset in instance_plan(True))
    dqn = build_dqn_command("python", 1000, run_id="dqn_x", resume=True,
                            instances=plan["DQN"][0], offset=plan["DQN"][1])
    ppo = build_ppo_command("python", 1000, run_id="ppo_x", resume=True,
                            instances=plan["PPO"][0], offset=plan["PPO"][1])
    assert option(dqn, "--instances") == "7" and option(dqn, "--offset") == "0"
    assert option(ppo, "--instances") == "7" and option(ppo, "--offset") == "7"
    assert "--resume" in dqn and "--resume" in ppo, "inherited means continue, not restart"


def run():
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return failed


if __name__ == "__main__":
    sys.exit(1 if run() else 0)
