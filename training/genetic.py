"""Evolve the hand-written controller's parameters on the emulators.

Aimed at the measured bottleneck. Wall contact runs 28-46% of frames even
for the current autopilot, contact is what drains power, and power is what
ends every run -- dead by about checkpoint 35 in the original analysis. The
controller's constants were tuned by racing four hand-picked configurations
against each other (tools/compare_configs.py). This searches the same space
properly.

It is also the cheapest way to improve everything else already built. The
autopilot is the behaviour-cloning teacher, so a better controller means a
better clone, which means a better starting policy for PPO:

    train_rl.bat evolve 8 20                    evolve, ~20 generations
    train_rl.bat teacher 8 25 --controller models\\controller_best.json
    train_rl.bat clone teacher_... bc_ppo
    train_rl.bat 8 500000 --resume --name bc_ppo

Why a GA rather than gradients: these parameters do not sit behind a
differentiable loss -- they feed an if-heavy controller whose output is a
button press, and the thing being scored is how far the machine gets over a
whole episode. There is no gradient to take. It is a small, bounded,
non-differentiable search, which is exactly what a GA is for.

Every genome is evaluated over several episodes, not one. That is not
caution, it is the house rule: identical configurations here have scored 81
then 66, and 96 then 54. A GA scoring single runs would spend its whole
budget selecting for luck and confidently converge on noise.
"""

import argparse
import json
import math
import os
import queue
import random
import statistics
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bridge_client import TOP_SPEED_RAW, BridgeTimeout
from control import SCAN_FRACTIONS, Controller
from env.fzero_env import DEFAULT_FRAME_SKIP, FZeroEnv
from training.run_store import RecordingWrapper, RunWriter, new_run_id
from vision.road_edges import measure_rows

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)

# name -> (low, high, integer?). Bounds are deliberately wider than the
# hand-tuned values, so the search can actually disagree with them, but not
# so wide that most of the population is nonsense.
GENES = {
    "deadzone": (0.005, 0.20, False),
    "full_scale": (0.10, 0.90, False),
    "lean_threshold": (0.10, 0.90, False),
    "curve_slowdown": (0.50, 4.00, False),
    "min_corner_speed": (400.0, 1600.0, False),
    "smoothing": (0.05, 1.00, False),
    "reversal_hold": (0.0, 24.0, True),
    "target_clearance": (25.0, 120.0, False),
    "row_near": (0.20, 1.50, False),
    "row_mid": (0.05, 1.50, False),
    "row_far": (0.05, 1.50, False),
}

# The current hand-tuned settings, seeded into the first population so the
# search starts from something that works and has to beat it.
SEED = {
    "deadzone": 0.04,
    "full_scale": 0.35,
    "lean_threshold": 0.30,
    "curve_slowdown": 1.9,
    "min_corner_speed": 800.0,
    "smoothing": 0.25,
    "reversal_hold": 8.0,
    "target_clearance": 55.0,
    "row_near": 1.0,
    "row_mid": 0.6,
    "row_far": 0.35,
}

DEFAULT_POPULATION = 12
DEFAULT_GENERATIONS = 20
DEFAULT_REPEATS = 3

ELITES = 2
TOURNAMENT = 3

# How many of the final population get a proper confirmation run, and how
# many episodes each. Cheap -- it happens once, not per generation.
FINALISTS = 3
DEFAULT_FINAL_REPEATS = 8
MUTATION_RATE = 0.25

# Mutation step, as a fraction of each gene's range. Large enough to escape
# a local optimum, small enough that a good genome is not destroyed.
MUTATION_SIGMA = 0.15

# A completed lap is the goal and nothing else comes close, so it dominates
# the fitness outright rather than competing with distance.
LAP_BONUS = 200.0

# How much a genome pays, in checkpoints of distance, for time spent in
# contact with a wall.
#
# This lives HERE and not in env/fzero_env.py's PENALTY_WALL_CONTACT, because
# that reward never reaches the genetic search: neuroevolution.py ranks
# genomes through summarise()/rank_key() on distance, laps, speed, turn
# quality and centring, and never reads the env reward at all. Raising the
# env penalty changes DQN and PPO only.
#
# Lowered from 40 because its job has moved. It was sized to tax wall-grinding
# -- in time-trial mode contact did not end an episode, so a genome could
# wedge against a barrier and bank the whole trial. WALL_CONTACT_FRAMES_LIMIT
# is now 3, so contact ends the run outright and grinding is no longer
# possible; the penalty no longer has to price a behaviour that cannot happen.
#
# At 40 it had also stopped being a tie-breaker and become the objective.
# Measured on a live generation: wall contact 69% of frames, so the term was
# worth -28 checkpoints against a mean distance of 26 -- mean fitness went
# negative and selection was effectively ranking "touched walls least" rather
# than "got furthest".
#
# At 10, and with episodes now ending on contact, a typical wall_rate of a few
# percent costs well under a checkpoint, so a clean line still edges out a
# scrapey one over equal ground without overriding distance. If wall_rate
# climbs back into the tens of percent this is the wrong knob -- it means
# contact-termination is not firing, so check WALL_CONTACT_FRAMES_LIMIT and
# --end-on-wall first.
WALL_RATE_PENALTY = 10.0

# Checkpoint-equivalent cost for each raw unit of average speed lost per
# emulated frame. The GA never ranks environment reward, so without this
# explicit term a large speed-decrease punishment in FZeroEnv would have no
# effect on which neural genomes reproduce. At 5.0, an average loss of two
# raw units per frame costs ten checkpoints of progress score.
SPEED_LOSS_RATE_PENALTY = 5.0

# Live map-independent ray behavior contributes a checkpoint-scale term.
# Correctly steering away from the closer fan scores +1, steering into it
# scores -1, and straight is neutral. Clearance recovery is separately
# credited so a policy must improve geometry, not merely press the expected
# button while continuing toward the wall.
#
# THE TWO WEIGHTS LOOK WILDLY DIFFERENT BECAUSE THE TWO SIGNALS ARE ON WILDLY
# DIFFERENT SCALES. Measured over 1163 episodes:
#
#   signal                        mean     p10      p90    natural spread
#   ray_response_score           0.024  -0.508   +0.356        0.86
#   mean_ray_clearance_recovery  0.013   0.007    0.020        0.013
#
# so a weight of 5 gave ray_response a 4.3-checkpoint swing while a weight of
# 10 gave clearance recovery 0.13 -- a tenth of a checkpoint, against a
# distance spread of 0 to 23. Clearance recovery was not a weak term, it was
# an invisible one, and no amount of tuning near 10 would have changed that.
#
# Sized deliberately rather than by round numbers: response now swings ~10.3
# checkpoints and clearance ~3.3, against distance's 23. Do NOT "correct" the
# 250 downward on the grounds that it looks large -- it is large because the
# quantity it multiplies is ~0.013, and lowering it silently deletes the term.
#
# The risk of going further: these are geometry proxies, not the goal. Push
# them much past this and the search can win by holding a tidy line while
# covering less ground, which is precisely the substitution that
# WALL_RATE_PENALTY at 40 caused. Watch that the best genome's distance keeps
# rising, not just its ray scores.
RAY_RESPONSE_PROGRESS_BONUS = 12.0
RAY_CLEARANCE_RECOVERY_BONUS = 250.0

# Selection is lexicographic at checkpoint resolution: lap completion, whole
# signed checkpoints, then a combined driving-quality score. See rank_key().
CENTRE_BONUS = 0.01

# Checkpoint-equivalent credit for average pace, added to progress_score so it
# actually selects.
#
# It used to be worth 0.0001, purely decorative, and rank_key()'s `speed` entry
# sat behind progress_score -- a continuous float that essentially never ties --
# so it never broke a comparison either. Between them, nothing in the search
# preferred a fast line to a slow one covering the same ground.
#
# `speed` here is the episode MEAN of speed/TOP_SPEED_RAW, in [0, 1]. A mean is
# length-independent, so this is NOT the farmable per-step speed payout that
# env/fzero_env.py documents at length: dawdling for more steps cannot bank it
# more times.
#
# Sizing: it is added alongside mean distance, which currently runs 13-50
# checkpoints. At 15.0 the full 0->1 speed range is worth 15, and the realistic
# 0.35-0.60 band is a ~3.75 checkpoint swing -- enough to separate two genomes
# that reach the same place, not enough for a genome to win by being quick and
# going nowhere. Raising it far above the typical distance spread would invert
# that, so re-check against mean distance before increasing it.
SPEED_PROGRESS_BONUS = 15.0

# Signed distance is measured in checkpoints. Rank it in one-checkpoint bands
# so gaining a whole checkpoint always wins, while the noisy fractional mean
# within a band can be decided by driving quality instead of an effectively
# unique float. This makes the documented secondary objectives real without
# allowing a clean stationary genome to beat one that travelled meaningfully.
DISTANCE_RANK_BAND = 1.0
TURN_QUALITY_RANK_BONUS = 5.0
CENTRE_RANK_BONUS = 3.0


def random_genome(rng):
    return {
        name: (
            float(rng.randint(int(low), int(high)))
            if is_int
            else rng.uniform(low, high)
        )
        for name, (low, high, is_int) in GENES.items()
    }


def clamp(genome):
    for name, (low, high, is_int) in GENES.items():
        value = min(max(genome[name], low), high)
        genome[name] = float(round(value)) if is_int else value
    return genome


def to_controller(genome):
    return Controller(
        deadzone=genome["deadzone"],
        full_scale=genome["full_scale"],
        lean_threshold=genome["lean_threshold"],
        curve_slowdown=genome["curve_slowdown"],
        min_corner_speed=genome["min_corner_speed"],
        smoothing=genome["smoothing"],
        reversal_hold=int(genome["reversal_hold"]),
        row_weights=(genome["row_near"], genome["row_mid"], genome["row_far"]),
        target_clearance=genome["target_clearance"],
    )


def run_episode(environment, recorder, genome):
    """Drive one episode with this genome's controller."""
    _, info = recorder.reset()
    controller = to_controller(genome)
    speed = info.get("speed", 0)
    wall_steps = 0
    centre_total = 0.0
    speed_total = 0.0
    risk_total = 0.0
    steps = 0

    while True:
        frame = environment.last_frame
        if frame is None:
            action = controller.update([], 128.0, speed)[0]
        else:
            height, width = frame.shape[:2]
            readings = measure_rows(frame, [int(height * f) for f in SCAN_FRACTIONS])
            action = controller.update(readings, width / 2.0, speed)[0]

        # The old controller returns steer/throttle/shoulder. The environment
        # now has no brake or lean action, so preserve only its steering
        # choice. This keeps this legacy parameter search safe to run while
        # the direct neural GA replaces it as the normal training route.
        # The legacy controller returns steer/throttle/shoulder. Coasting is
        # removed from the environment, but keep its lean choice.
        _, _, terminated, truncated, info = recorder.step((action[0], action[2]))
        speed = info.get("speed", 0)
        steps += 1
        wall_steps += bool(info.get("wall"))
        centre_total += 1.0 - min(max(float(info.get("off_centre", 1.0)), 0.0), 1.0)
        speed_total += min(max(float(info.get("speed", 0)) / TOP_SPEED_RAW, 0.0), 1.0)
        risk_total += min(max(float(info.get("wall_risk", 0.0)), 0.0), 1.0)

        if terminated or truncated:
            break

    return {
        "furthest": info.get("furthest", 0),
        "wall_rate": wall_steps / max(steps, 1),
        "centre": 1.0 - float(
            info.get("mean_off_centre", 1.0 - centre_total / max(steps, 1))
        ),
        "speed": float(
            info.get("mean_speed_normalised", speed_total / max(steps, 1))
        ),
        "speed_loss": float(info.get("mean_speed_loss", 0.0)),
        "ray_response": float(info.get("ray_response_score", 0.0)),
        "ray_clearance_recovery": float(
            info.get("mean_ray_clearance_recovery", 0.0)
        ),
        "wall_risk": float(
            info.get("mean_wall_risk", risk_total / max(steps, 1))
        ),
        "safety_override_rate": float(info.get("safety_override_rate", 0.0)),
        "actual_lean_rate": float(info.get("actual_lean_rate", 0.0)),
        "lap": int(bool(info.get("finished"))),
    }


DEAD_EPISODE = {
    "furthest": 0,
    "wall_rate": 1.0,
    "centre": 0.0,
    "speed": 0.0,
    "speed_loss": 0.0,
    "ray_response": -1.0,
    "ray_clearance_recovery": 0.0,
    "wall_risk": 1.0,
    "safety_override_rate": 0.0,
    "policy_agreement": 0.0,
    "actual_lean_rate": 0.0,
    "frontier_clearance": 0.0,
    "frontier_centre": 0.0,
    "turn_recovery": 0.0,
    "safe_frontier_rate": 0.0,
    "turn_quality": 0.0,
    "curriculum": 0,
    "lap": 0,
}

# Episodes kept per genome when its evidence accumulates across generations
# (see History). Three generations' worth: enough to sharpen the estimate a
# lot, few enough that a genome is still judged on recent behaviour.
MAX_HISTORY = 9


def summarise(episodes):
    """Fitness for one genome from its episodes."""
    distances = [e.get("distance", e["furthest"]) for e in episodes]
    furthest_distances = [e["furthest"] for e in episodes]
    laps = sum(e["lap"] for e in episodes)
    centre = statistics.mean(e.get("centre", 0.0) for e in episodes)
    speed = statistics.mean(e.get("speed", 0.0) for e in episodes)
    speed_loss = statistics.mean(e.get("speed_loss", 0.0) for e in episodes)
    ray_response = statistics.mean(e.get("ray_response", 0.0) for e in episodes)
    ray_clearance_recovery = statistics.mean(
        e.get("ray_clearance_recovery", 0.0) for e in episodes
    )
    wall_risk = statistics.mean(e.get("wall_risk", 0.0) for e in episodes)
    safety_override_rate = statistics.mean(
        e.get("safety_override_rate", 0.0) for e in episodes
    )
    actual_lean_rate = statistics.mean(
        e.get("actual_lean_rate", 0.0) for e in episodes
    )
    policy_agreement = statistics.mean(
        e.get("policy_agreement", 0.0) for e in episodes
    )
    frontier_clearance = statistics.mean(
        e.get("frontier_clearance", 0.0) for e in episodes
    )
    frontier_centre = statistics.mean(
        e.get("frontier_centre", 0.0) for e in episodes
    )
    turn_recovery = statistics.mean(
        e.get("turn_recovery", 0.0) for e in episodes
    )
    safe_frontier_rate = statistics.mean(
        e.get("safe_frontier_rate", 0.0) for e in episodes
    )
    turn_quality = statistics.mean(
        e.get("turn_quality", 0.0) for e in episodes
    )
    lap_rate = laps / len(episodes)
    curriculum_rate = statistics.mean(e.get("curriculum", 0.0) for e in episodes)

    return {
        "fitness": (
            statistics.mean(distances)
            + LAP_BONUS * lap_rate
            + CENTRE_BONUS * centre
            + SPEED_PROGRESS_BONUS * speed
            - SPEED_LOSS_RATE_PENALTY * speed_loss
            + RAY_RESPONSE_PROGRESS_BONUS * ray_response
            + RAY_CLEARANCE_RECOVERY_BONUS * ray_clearance_recovery
        ),
        "distance": statistics.mean(distances),
        # Distance actually earned, after paying for time spent scrubbing a
        # wall. rank_key selects on THIS rather than raw distance, so grinding
        # along a barrier can no longer buy progress cheaply.
        "progress_score": (
            statistics.mean(distances)
            - WALL_RATE_PENALTY * statistics.mean(e["wall_rate"] for e in episodes)
            + SPEED_PROGRESS_BONUS * speed
            - SPEED_LOSS_RATE_PENALTY * speed_loss
            + RAY_RESPONSE_PROGRESS_BONUS * ray_response
            + RAY_CLEARANCE_RECOVERY_BONUS * ray_clearance_recovery
        ),
        "furthest": statistics.mean(furthest_distances),
        "spread": statistics.stdev(distances) if len(distances) > 1 else 0.0,
        "wall_rate": statistics.mean(e["wall_rate"] for e in episodes),
        "centre": centre,
        "speed": speed,
        "speed_loss": speed_loss,
        "ray_response": ray_response,
        "ray_clearance_recovery": ray_clearance_recovery,
        "wall_risk": wall_risk,
        "safety_override_rate": safety_override_rate,
        "policy_agreement": policy_agreement,
        "actual_lean_rate": actual_lean_rate,
        "frontier_clearance": frontier_clearance,
        "frontier_centre": frontier_centre,
        "turn_recovery": turn_recovery,
        "safe_frontier_rate": safe_frontier_rate,
        "turn_quality": turn_quality,
        "lap_rate": lap_rate,
        "laps": laps,
        "episodes": len(episodes),
        "curriculum_rate": curriculum_rate,
    }


def rank_key(score):
    """Completion, whole signed checkpoints, then combined driving quality.

    Turn quality is geometry-based rather than button-based -- it scores
    clearance, centring, risk-recovery and safe frontier movement, and none
    of those terms reference the lean button directly, so a genome cannot
    earn it by merely holding a turn direction.

    But that only stops the NAIVEST version of the problem. turn_quality
    grades OUTCOMES, not the cost of achieving them: a genome that holds
    lean through corners AND straights alike can still post good clearance
    and centring, since lean is genuinely the tool for tight cornering here.
    Nothing was checking whether the lean was actually warranted.

    speed is what should have caught that -- lean scrubs speed whether or
    not a corner needed it -- but it sat LAST in this tuple, behind three
    terms (turn_quality, policy_agreement, centre) that a converged
    population tends to tie or nearly tie on. It essentially never got to
    break a comparison. Measured: the fraction of episodes spent holding
    lean for ~100% of the episode climbed from ~15% to over 50% across one
    run, with no matching gain in furthest progress -- fitness could not
    tell a genome that leaned only where needed from one that leaned
    everywhere, provided both happened to reach similar ground.
    The former tuple put geometry behind continuous floats, so it almost never
    participated. One-checkpoint bands preserve meaningful progress while the
    combined quality score lets wall contact, pace, rays, turning and centering
    jointly decide fractional near-ties.
    """
    distance = float(score.get("distance", score["furthest"]))
    distance_band = math.floor(distance / DISTANCE_RANK_BAND)
    # progress_score already contains wall contact, pace, speed loss, live ray
    # response, clearance recovery and the deterministic ray-probe adjustment.
    # Remove raw distance because its strict priority is represented by the
    # band above, then add the two geometry outcomes that previously sat behind
    # continuous values and consequently never affected selection.
    quality_score = (
        float(score.get("progress_score", distance))
        - distance
        + TURN_QUALITY_RANK_BONUS * float(score.get("turn_quality", 0.0))
        + CENTRE_RANK_BONUS * float(score.get("centre", 0.0))
    )
    return (
        score.get("lap_rate", 0.0),
        distance_band,
        quality_score,
        distance,
        score.get("ray_probe_score", score.get("ray_response", 0.0)),
        score.get("ray_response", 0.0),
        score.get("speed", 0.0),
        score.get("turn_quality", 0.0),
        score.get("policy_agreement", 0.0),
        score.get("centre", 0.0),
    )


class History:
    """Evidence about a genome, pooled across the generations it survives.

    Elitism carries the best genomes forward unchanged, and they are
    re-evaluated every generation anyway. Throwing the previous episodes
    away means a genome that got lucky once is judged on three fresh
    episodes and can stay lucky; pooling them means a survivor's estimate
    gets sharper the longer it survives, and a fluke is corrected downward.

    This costs nothing -- the episodes are being run either way. It is the
    cheapest defence available against the failure this search is most prone
    to, which is selecting for luck: identical configurations here have
    scored 81 then 66.

    Newcomers are still judged on a single generation's episodes, so they
    can out-rank a well-measured elite by luck. Elitism is what stops that
    from losing the elite.
    """

    def __init__(self, max_history=MAX_HISTORY):
        self.max_history = max_history
        self._episodes = {}

    @staticmethod
    def key(genome):
        # Rounded, so floating-point noise cannot make a carried-forward
        # elite look like a different genome.
        return tuple(round(genome[name], 6) for name in sorted(GENES))

    def add(self, genome, episodes):
        key = self.key(genome)
        pooled = self._episodes.get(key, []) + list(episodes)
        self._episodes[key] = pooled[-self.max_history :]
        return self._episodes[key]


def worker(context, jobs, episodes):
    """One emulator, pulling single episodes off the queue until it is empty.

    The queue holds EPISODES, not genomes. Handing out a whole genome at a
    time packs badly: 12 genomes of 3 episodes across 8 emulators runs 8 in
    parallel and then 4, leaving half the emulators idle for the second
    round -- 6 episode-times of wall clock for 4.5 episode-times of work.
    Splitting to the episode evens it out for nothing.
    """
    instance, run_id, frame_skip = context
    environment = FZeroEnv(instance=instance, frame_skip=frame_skip)
    writer = RunWriter(
        run_id,
        instance=instance,
        meta={"source": "genetic", "frame_skip": frame_skip},
    )
    recorder = RecordingWrapper(environment, writer)

    try:
        while True:
            try:
                genome_index, repeat_index, genome = jobs.get_nowait()
            except queue.Empty:
                return

            try:
                # Each slot is written by exactly one thread, so no lock.
                episodes[genome_index][repeat_index] = run_episode(
                    environment, recorder, genome
                )
            except BridgeTimeout as exc:
                print(f"  instance {instance}: {exc}")
                episodes[genome_index][repeat_index] = dict(DEAD_EPISODE)
            finally:
                jobs.task_done()
    except Exception as exc:  # noqa: BLE001 - one bad emulator must not stop the search
        print(f"  instance {instance}: {type(exc).__name__}: {exc}")
    finally:
        recorder.close()


def score_population(population, instances, run_id, repeats, frame_skip, history=None,
                     offset=0):
    jobs = queue.Queue()

    # Interleaved by repeat, so if an emulator dies partway every genome has
    # had a first episode rather than the early ones having three and the
    # late ones none.
    for repeat_index in range(repeats):
        for genome_index, genome in enumerate(population):
            jobs.put((genome_index, repeat_index, genome))

    episodes = [[None] * repeats for _ in population]

    threads = []
    for instance in range(offset, offset + instances):
        thread = threading.Thread(
            target=worker,
            args=((instance, run_id, frame_skip), jobs, episodes),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    # An emulator that died leaves holes; count those episodes as worthless
    # rather than letting a None reach the arithmetic.
    scores = []
    for genome, genome_episodes in zip(population, episodes):
        fresh = [e or dict(DEAD_EPISODE) for e in genome_episodes]
        pooled = history.add(genome, fresh) if history is not None else fresh
        scores.append(summarise(pooled))

    return scores


def tournament(population, scores, rng, size=TOURNAMENT):
    """Pick the best of `size` random contenders, by rank_key order.

    size is a parameter rather than the module constant because this is
    shared with training/neuroevolution.py, which runs a much larger
    population and wants stronger selection pressure. Defaulting to
    TOURNAMENT keeps the legacy controller search exactly as it was.
    """
    contenders = rng.sample(range(len(population)), min(size, len(population)))
    best = max(contenders, key=lambda i: rank_key(scores[i]))
    return population[best]


def breed(parent_a, parent_b, rng):
    """Uniform crossover, then gaussian mutation on a fraction of genes."""
    child = {
        name: (parent_a[name] if rng.random() < 0.5 else parent_b[name])
        for name in GENES
    }

    for name, (low, high, _) in GENES.items():
        if rng.random() < MUTATION_RATE:
            child[name] += rng.gauss(0.0, (high - low) * MUTATION_SIGMA)

    return clamp(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0,
                        help="first instance number to use -- set this to "
                             "run alongside another trainer on a disjoint "
                             "range, e.g. DQN on 0-7 and this on 8+")
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="episodes per genome -- do not set this to 1")
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--final-repeats", type=int, default=DEFAULT_FINAL_REPEATS,
                        help="episodes per finalist in the confirmation run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "controller_best.json"))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    from training.session_lock import EmulatorSessionLock

    with EmulatorSessionLock(
        "controller genetic search", offset=args.offset, count=max(args.instances, 1)
    ):
        _run(args)


def _run(args):
    if args.repeats < 2:
        print("! --repeats 1 selects for luck: identical configs here have "
              "scored 81 then 66. Continuing, but do not trust the result.\n")

    rng = random.Random(args.seed)
    run_id = args.run_id or new_run_id("evolve")

    # Seeded with the current settings plus a lightly mutated copy, so the
    # search must beat what is already there rather than rediscovering it.
    population = [dict(SEED), breed(SEED, SEED, rng)]
    population += [random_genome(rng) for _ in range(args.population - 2)]

    print(f"population {args.population}, {args.generations} generations, "
          f"{args.repeats} episodes per genome")
    print(f"{args.population * args.repeats} episodes per generation across "
          f"{args.instances} emulators")
    print(f"run '{run_id}'\n")

    best = None
    best_score = None
    started = time.time()
    history = History()

    for generation in range(args.generations):
        scores = score_population(
            population, args.instances, run_id, args.repeats, args.frame_skip,
            history, offset=args.offset,
        )

        ranked = sorted(
            zip(population, scores), key=lambda pair: rank_key(pair[1]), reverse=True
        )
        leader, leader_score = ranked[0]

        if best_score is None or rank_key(leader_score) > rank_key(best_score):
            best, best_score = dict(leader), dict(leader_score)
            with open(args.out, "w") as fh:
                json.dump({"genome": best, "score": best_score}, fh, indent=2)

        mean_fitness = statistics.mean(s["fitness"] for s in scores)
        print(
            f"gen {generation + 1:3d}/{args.generations}  "
            f"best {leader_score['fitness']:6.1f} "
            f"(cp {leader_score['furthest']:5.1f} +-{leader_score['spread']:4.1f} "
            f"over {leader_score['episodes']:2d} eps, "
            f"wall {leader_score['wall_rate'] * 100:4.1f}%, "
            f"laps {leader_score['laps']})   "
            f"mean {mean_fitness:6.1f}   "
            f"[{(time.time() - started) / 60:.1f} min]",
            flush=True,
        )

        final_ranked = ranked

        if generation == args.generations - 1:
            break

        # Elitism, then fill the rest by tournament selection and breeding.
        survivors = [dict(genome) for genome, _ in ranked[:ELITES]]
        while len(survivors) < args.population:
            survivors.append(
                breed(
                    tournament(population, scores, rng),
                    tournament(population, scores, rng),
                    rng,
                )
            )
        population = survivors

    # Confirmation run. Picking the winner by best-fitness-ever is the
    # winner's curse in its purest form: with a handful of episodes each, the
    # genome that comes top is disproportionately the one that got lucky, and
    # the margin it won by is mostly noise. Observed directly during
    # development -- a genome scored 76.0 over 2 episodes and 73.2 once it had
    # 6, and it was still the best one.
    #
    # So the top few are re-run properly and the winner decided on that.
    finalists = [genome for genome, _ in final_ranked[:FINALISTS]]
    print(f"\nConfirming the top {len(finalists)} on "
          f"{args.final_repeats} fresh episodes each...")

    final_scores = score_population(
        finalists, args.instances, run_id, args.final_repeats, args.frame_skip,
        offset=args.offset,
    )

    confirmed = sorted(zip(finalists, final_scores), key=lambda pair: rank_key(pair[1]), reverse=True)
    for rank, (_, score) in enumerate(confirmed, 1):
        print(f"  {rank}. cp {score['furthest']:5.1f} +-{score['spread']:4.1f}  "
              f"wall {score['wall_rate'] * 100:4.1f}%  laps {score['laps']}"
              f"/{score['episodes']}")

    best, best_score = dict(confirmed[0][0]), dict(confirmed[0][1])
    with open(args.out, "w") as fh:
        json.dump(
            {"genome": best, "score": best_score, "confirmed": True},
            fh,
            indent=2,
        )

    print(f"\nbest fitness {best_score['fitness']:.1f} "
          f"(checkpoint {best_score['furthest']:.1f} +-{best_score['spread']:.1f} "
          f"over {best_score['episodes']} confirmation episodes, "
          f"wall {best_score['wall_rate'] * 100:.1f}%)")
    print(f"seed for comparison: the hand-tuned settings are the first genome "
          f"of generation 1.\n")
    for name in GENES:
        print(f"  {name:18s} {best[name]:8.3f}   (was {SEED[name]:8.3f})")

    print(f"\nWritten to {args.out}")
    print("Use it as the teacher:\n"
          f"    .venv\\Scripts\\python.exe training\\record_teacher.py "
          f"--instances 8 --episodes 25 --controller {args.out}")


if __name__ == "__main__":
    main()
