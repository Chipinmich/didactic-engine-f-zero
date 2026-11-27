"""Evolve the driving policy directly, in place of PPO.

Same job PPO was doing -- map the stacked observation to steering and lean choices --
but searched with a genetic algorithm instead of gradients.

Why this is worth trying here, on the evidence:

  PPO has never learned anything on this task, and the reasons are all about
  gradients. Returns ran to thousands and had to be fitted by a value head;
  the advantage signal was noisy enough that pairwise concordance between
  progress and reward measured only 64%; and the runs were starved of
  updates. A GA needs none of that. It never estimates a value function,
  never computes an advantage, and does not care that the reward is badly
  scaled -- it only needs to rank whole episodes against each other, which
  is the one thing this env does reliably.

  The reward can also stay exactly as it is. A GA is indifferent to a
  constant offset, so "every episode scores negative" stops being a problem.

  And it parallelises the way this project already works: whole episodes on
  separate emulators, no shared gradient.

V4 uses a 32-unit tanh hidden layer.  Seven rays describe useful geometric
relationships (a closing side fan, an opening opposite side, and their
change over time) that a linear policy could not combine reliably.  The
larger search is seeded by lifting the balanced linear teacher into the MLP,
then protected with coherent hidden-unit crossover and behavioural diversity.

    train_rl.bat evolve-policy
    python training/neuroevolution.py --instances 8 --generations 60

The winner is saved to models/neuro_best.npz and can be driven with
tools/evaluate.py --neuro.
"""

import argparse
import hashlib
import os
import pickle
import queue
import random
import statistics
import sys
import threading
import time

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from bridge_client import (
    CHECKPOINTS_PER_LAP,
    STEER_LEFT,
    STEER_RIGHT,
    STEER_STRAIGHT,
    TOP_SPEED_RAW,
    BridgeTimeout,
)
from env.fzero_env import (
    DEFAULT_FRAME_SKIP,
    MAX_EPISODE_FRAMES,
    WALL_CONTACT_FRAMES_LIMIT,
    FZeroEnv,
)
# Imported as a module as well as by name, so --speed-weight can rebind
# SPEED_PROGRESS_BONUS where summarise() actually reads it. Rebinding a
# from-import here would only change this module's copy.
from training import genetic
from training.genetic import (
    DEAD_EPISODE,
    ELITES,
    History,
    rank_key,
    summarise,
    tournament,
)
from training.run_store import (
    RecordingWrapper,
    RunWriter,
    append_generation_summary,
    list_runs,
    load_episodes,
    load_meta,
    load_steps,
    new_run_id,
)
from vision.road_edges import (
    LEGACY_NUM_FEATURES,
    NUM_FEATURES,
    NUM_RAY_FEATURES,
    RAY_START_INDEX,
    ROAD_LEFT_INDEX,
    ROAD_RIGHT_INDEX,
    SPEED_INDEX,
)

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)

DEFAULT_FRAME_STACK = 8
DEFAULT_POPULATION = 48
DEFAULT_GENERATIONS = 60
# Was 3. Evaluation noise, not diversity, is what stalls this search:
# per-genome spread ran +-6 to +-13 checkpoints on means around 15, so three
# episodes cannot separate genomes whose true means differ by less than
# roughly 8 -- which is most of them. Six roughly halves the standard error
# of each genome's estimate.
#
# This costs real time: at 48 genomes it doubles the episodes per generation
# (staged screening via --screen-repeats claws some back by spending the
# extra episodes only on genomes that survive the first cut). Lower it to 5
# if generations are running too slowly to watch.
DEFAULT_REPEATS = 6
DEFAULT_FINAL_REPEATS = 12
DEFAULT_SCREEN_REPEATS = 2
DEFAULT_SCREEN_FRACTION = 0.75
DEFAULT_CURRICULUM_START = 20
DEFAULT_CURRICULUM_STEP = 5
CHECKPOINT_VERSION = 1
EVOLUTION_RULES_VERSION = 9
EMULATED_FPS = 60
DEFAULT_EVALUATION_SECONDS = 30.0
DEFAULT_HIDDEN = 32
POLICY_FINALISTS = 5
HALL_OF_FAME_SIZE = 8
HALL_OF_FAME_PER_GENERATION = 2

# The policy chooses one of 3 steering directions and one of 3 lean states.
# Acceleration is held by FZeroEnv, so no policy action can coast/brake.
NUM_ACTIONS = 9


def genome_id(genome):
    """Stable short identity for episode, parent and generation records."""
    payload = np.asarray(genome, dtype=np.float32).tobytes()
    return hashlib.sha256(payload).hexdigest()[:12]


def default_lineage(population, source="initial"):
    return [
        {"genome_id": genome_id(genome), "source": source, "parent_ids": []}
        for genome in population
    ]


def generation_summary(generation, population, scores, lineage):
    """Serializable ranking and ancestry for one completed generation."""
    order = sorted(range(len(population)), key=lambda i: rank_key(scores[i]), reverse=True)
    ranks = {index: rank for rank, index in enumerate(order, 1)}
    lineage = lineage if len(lineage or []) == len(population) else default_lineage(
        population, "unknown"
    )
    score_fields = (
        "fitness", "progress_score", "distance", "furthest", "spread",
        "wall_rate", "centre", "speed", "speed_loss", "ray_probe_score",
        "ray_response", "ray_clearance_recovery", "turn_quality",
        "policy_agreement", "actual_lean_rate", "safety_override_rate",
        "lap_rate", "laps", "episodes",
    )
    genomes = []
    for index, (genome, score) in enumerate(zip(population, scores)):
        ancestry = lineage[index]
        genomes.append({
            "index": index,
            "genome_id": genome_id(genome),
            "rank": ranks[index],
            "source": ancestry.get("source", "unknown"),
            "parent_ids": list(ancestry.get("parent_ids", [])),
            "score": {
                key: float(score[key])
                for key in score_fields
                if key in score
            },
        })
    return {
        "generation": int(generation),
        "population_size": len(population),
        "genomes": genomes,
    }

# Each variant advances exactly four fully accelerated frames before episode
# accounting starts. Equal length avoids handing one scenario free speed or
# progress. Left/right variants are mirrored, and no variant uses lean or
# coasting. The generation schedule below rotates these real nearby states
# across slots while act() itself remains deterministic.
START_VARIANTS = (
    (STEER_STRAIGHT, STEER_STRAIGHT, STEER_STRAIGHT, STEER_STRAIGHT),
    (STEER_LEFT, STEER_LEFT, STEER_STRAIGHT, STEER_STRAIGHT),
    (STEER_RIGHT, STEER_RIGHT, STEER_STRAIGHT, STEER_STRAIGHT),
    (STEER_STRAIGHT, STEER_STRAIGHT, STEER_LEFT, STEER_LEFT),
    (STEER_STRAIGHT, STEER_STRAIGHT, STEER_RIGHT, STEER_RIGHT),
)

# Mutation step on the weights. Larger than it looks: the inputs are all in
# [0, 1] and there are many stacked inputs, so a small per-weight change moves the
# logits meaningfully.
MUTATION_SIGMA = 0.05
MUTATION_RATE = 0.20
# Exploratory children still start from proven parents.  At 0.20 their
# measured mean distance collapsed well below ordinary bred children, so use
# a moderate step that can change ray/turn behaviour without erasing it.
LARGE_MUTATION_SIGMA = 0.10
STALL_GENERATIONS = 5

# Stronger than genetic.py's tournament of 3. Over a population of 48, a
# tournament of 3 gives weak pressure -- a below-median genome still wins
# its tournament fairly often. 4 sharpens selection without collapsing the
# population onto a single lineage.
TOURNAMENT_SIZE = 4

# Bias mutation is deliberately stronger than weight mutation, because a
# bias shifts a whole action's logit regardless of input and is the fastest
# way to change behaviour. It was running at 2x (hidden) and 3x (output) the
# weight sigma, which at the old stalled sigma of 0.30 meant output biases
# jumping by ~0.9 in a single step -- the same magnitude that previously
# collapsed the policy onto one constant action. Capped to 1.5-2x.
HIDDEN_BIAS_SIGMA_SCALE = 1.5
OUTPUT_BIAS_SIGMA_SCALE = 2.0

# What happens on a plateau. The old response was a single wider mutation
# tier (STALLED_MUTATION_SIGMA = 0.30) applied to everyone, which jitters
# the whole population around the same local optimum -- measured, 60
# generations flat with that tier active for nearly all of them.
#
# Instead: a partial mass extinction. The weakest half is wiped and replaced
# with fresh random genomes, while elites and the behavioural-diversity picks
# survive untouched. That restarts most lineages from scratch rather than
# perturbing every one of them, which is the difference between searching
# somewhere new and searching the same basin harder.
EXTINCTION_FRACTION = 0.50
LARGE_MUTANT_FRACTION = 0.08
RANDOM_IMMIGRANT_FRACTION = 0.20
ELITE_VARIANT_FRACTION = 0.10
DIVERSITY_ELITES = 2
DIVERSITY_PROBE_COUNT = 256
GUIDANCE_VARIANTS = 2
PARENT_POOL_SIZE = 16

# Initial weights. Small, so the first generation is near-indifferent between
# the three actions and the search opens up the differences itself rather
# than starting from a strong arbitrary bias.
INIT_SCALE = 0.15

# Random immigrants are born biased TOWARD turning, relative to whatever the
# current elite is biased toward.
#
# Why: the elite this was measured against had an output bias of +1.125 on
# action 0 (steer straight, no lean) -- about 7.5x INIT_SCALE -- while every
# one of the six turning actions sat at or below zero. Replayed on 1415 real
# recorded observations it chose straight 98.5% of the time and only ever
# used 5 of its 9 actions. A fresh immigrant drawn from N(0, INIT_SCALE) has
# no such bias, but it also has no reason to prefer turning, so the gene pool
# kept being refreshed with genomes that were equally happy to go straight --
# which is the behaviour the population had already collapsed onto.
#
# The boost is measured against the elite's own straight-vs-turn gap rather
# than being a fixed number, so it stays meaningful as that gap changes: a
# multiplier of 3.0 gives immigrants a turning preference 200% larger than
# the elite's current preference, in the opposite direction. On the measured
# elite (gap -0.347) that works out to about +1.04 on each turning action.
#
# It biases toward turning, NOT toward a direction: all six turning actions
# get the same boost, so left vs right is still decided by the weights.
TURN_IMMIGRANT_MULTIPLIER = 3.0

# Used before any elite exists (generation 1), and as an upper bound so a
# pathological elite cannot produce an immigrant that can only ever turn.
DEFAULT_TURN_IMMIGRANT_BIAS = 2.0 * INIT_SCALE
TURN_IMMIGRANT_CAP = 8.0 * INIT_SCALE


def turning_bias_for_immigrants(elite=None):
    """How much to favour turning in a freshly seeded genome.

    Returns a positive offset added to the six turning actions' output
    biases. Derived from the elite's own straight-vs-turn gap so it tracks
    whatever the population has currently converged onto.
    """
    if elite is None or len(elite) < NUM_ACTIONS:
        return DEFAULT_TURN_IMMIGRANT_BIAS

    output_bias = np.asarray(elite[-NUM_ACTIONS:], dtype=np.float64)
    straight = float(output_bias[:3].mean())
    turning = float(output_bias[3:].mean())
    gap = abs(turning - straight)
    return float(min(TURN_IMMIGRANT_MULTIPLIER * gap, TURN_IMMIGRANT_CAP))


# Steering mix a freshly drawn genome should start with, as
# (straight, left, right). Raised from 50/20/30 to bias harder toward
# turning: 65% of the opening tendency now steers. Right still outweighs
# left, since this section of track curves that way, but neither direction
# is suppressed -- a one-direction opening bias is how a previous lineage
# ended up at 2% left and unable to corner the other way.
TARGET_ACTION_MIX = (0.35, 0.25, 0.40)

# Per-steer-group offsets added to the output bias to produce that mix.
#
# Solved numerically against the REAL act() path rather than guessed, with
# straight as the fixed reference at 0.
#
# A first pass solved it assuming logits are just the output bias. That holds
# for a linear policy but NOT for the 32-unit MLP: on a blank observation the
# hidden layer still contributes w2 @ tanh(b1), so the bias is not the whole
# logit. Those offsets landed at 45.7/22.4/31.9. Re-solving through act()
# itself gives the values below.
#
# Measured on a held-out seed, hidden=32: straight 35.7%, left 24.8%,
# right 39.5% against the 35/25/40 target -- a 64.3% turning share against
# the 65% asked for.
#
# These are tied to INIT_SCALE, TARGET_ACTION_MIX *and* hidden size --
# re-solve if any of the three changes.
STEER_GROUP_BIAS = (0.0, -0.0392, +0.0129)

# Ray inputs start life with stronger connections than the row-geometry
# inputs. The seven rays are the newest and most directly useful geometry
# (a closing fan on one side, an opening one on the other), but they are 7
# of 22 features per frame, so a uniformly random genome barely listens to
# them. This is the same treatment emphasise_ray_connections applies to a
# forked seed, brought to freshly drawn genomes since a clean-lineage run
# has no fork to apply it to.
RAY_WEIGHT_GAIN = 2.0
RAY_MUTATION_RATE_GAIN = 2.0
RAY_MUTATION_SIGMA_GAIN = 1.5
RAY_PROBE_PROGRESS_BONUS = 6.0


def ray_input_mask(inputs):
    """Boolean mask selecting all seven ray inputs in every stacked frame."""
    mask = np.zeros(int(inputs), dtype=bool)
    stack, remainder = divmod(int(inputs), NUM_FEATURES)
    if remainder:
        return mask
    for frame in range(stack):
        start = frame * NUM_FEATURES + RAY_START_INDEX
        mask[start : start + NUM_RAY_FEATURES] = True
    return mask


def synthetic_ray_probes(stack):
    """Controlled rays-only situations and the steering they require."""
    patterns = (
        ((0.10, 0.18, 0.32, 0.72, 0.86, 0.94, 1.00), STEER_RIGHT),
        ((1.00, 0.94, 0.86, 0.72, 0.32, 0.18, 0.10), STEER_LEFT),
        ((0.18, 0.28, 0.42, 0.35, 0.76, 0.90, 0.98), STEER_RIGHT),
        ((0.98, 0.90, 0.76, 0.35, 0.42, 0.28, 0.18), STEER_LEFT),
        ((0.92, 0.96, 1.00, 1.00, 1.00, 0.96, 0.92), STEER_STRAIGHT),
    )
    observations = []
    targets = []
    for scale in (0.9, 1.0, 1.1):
        for rays, target in patterns:
            one = np.zeros(NUM_FEATURES, dtype=np.float32)
            one[SPEED_INDEX] = 0.5
            one[RAY_START_INDEX : RAY_START_INDEX + NUM_RAY_FEATURES] = np.clip(
                np.asarray(rays, dtype=np.float32) * scale, 0.0, 1.0
            )
            observations.append(np.tile(one, int(stack)))
            targets.append(target)
    return np.asarray(observations, dtype=np.float32), np.asarray(targets, dtype=np.int64)


def policy_ray_probe_score(genome, stack, inputs, hidden):
    """Accuracy on deterministic left/right/clear ray situations."""
    if np.asarray(genome).size != genome_size(inputs, hidden):
        # Unit-test/mocked populations may use tiny identity vectors instead
        # of real policy weights. Neutral keeps those scheduling tests about
        # scheduling rather than coupling them to the network layout.
        return 1.0 / 3.0
    observations, targets = synthetic_ray_probes(stack)
    predicted_steer = action_signature(genome, observations, inputs, hidden) // 3
    return float(np.mean(predicted_steer == targets))


def add_ray_probe_fitness(population, scores, stack, inputs, hidden):
    """Attach deterministic ray understanding to every GA score."""
    for genome, score in zip(population, scores):
        probe = policy_ray_probe_score(genome, stack, inputs, hidden)
        adjustment = RAY_PROBE_PROGRESS_BONUS * (probe - (1.0 / 3.0))
        score["ray_probe_score"] = probe
        score["progress_score"] = score.get("progress_score", 0.0) + adjustment
        score["fitness"] = score.get("fitness", 0.0) + adjustment
    return scores


def random_immigrant(rng, size, turn_bias=None, inputs=None, hidden=None,
                     ray_gain=RAY_WEIGHT_GAIN):
    """A fresh random genome with a deliberate opening steering tendency.

    The output bias is the LAST NUM_ACTIONS entries in both genome layouts
    (linear: weights then bias; MLP: w1, b1, w2, b2). Action code is
    steer * 3 + lean, so entries 0..2 are straight, 3..5 left, 6..8 right --
    which is what STEER_GROUP_BIAS indexes.

    turn_bias is accepted for callers that still pass it and is applied to
    the six turning actions on top of the group offsets, so an elite-relative
    boost and a target mix can coexist.
    """
    genome = rng.normal(0.0, INIT_SCALE, size).astype(np.float32)

    if size >= NUM_ACTIONS:
        group_offsets = np.repeat(np.asarray(STEER_GROUP_BIAS, dtype=np.float32), 3)
        genome[-NUM_ACTIONS:] += group_offsets
        if turn_bias:
            genome[-(NUM_ACTIONS - 3):] += np.float32(turn_bias)

    # Ray emphasis needs the layer shapes; skip it when the caller cannot
    # supply them rather than guessing and corrupting the genome.
    if ray_gain and ray_gain != 1.0 and inputs is not None and hidden is not None:
        # The stack is implied by the input width, so callers do not have to
        # thread it through: inputs = stack * NUM_FEATURES by construction.
        stack, remainder = divmod(inputs, NUM_FEATURES)
        if remainder == 0 and stack > 0 and size == genome_size(inputs, hidden):
            genome = emphasise_ray_connections(genome, stack, inputs, hidden, ray_gain)

    return genome


def parse_start_slots(value):
    """Parse a comma-separated turn curriculum, e.g. ``2,3,4``."""
    if value is None or str(value).strip() == "":
        return ()
    slots = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not slots or any(slot < 0 or slot > 9 for slot in slots):
        raise argparse.ArgumentTypeError("start slots must be comma-separated values 0..9")
    return slots


class Stacker:
    """The frame stack the policy sees, for one emulator.

    Same layout as VecFrameStack: oldest first, newest last, zeros before the
    episode started. Pinned by tests/test_run_store.py.
    """

    def __init__(self, stack, features=NUM_FEATURES):
        self.stack = stack
        self.features = features
        self.buffer = np.zeros(stack * features, dtype=np.float32)

    def reset(self, observation):
        self.buffer[:] = 0.0
        self.buffer[-self.features :] = observation
        return self.buffer

    def push(self, observation):
        self.buffer[: -self.features] = self.buffer[self.features :]
        self.buffer[-self.features :] = observation
        return self.buffer


def stack_episode(observations, stack):
    """Build the same oldest-to-newest stack used by the live policy."""
    steps, features = observations.shape
    out = np.zeros((steps, features * stack), dtype=np.float32)
    for offset in range(stack):
        age = stack - 1 - offset
        if age == 0:
            out[:, offset * features : (offset + 1) * features] = observations
        else:
            out[age:, offset * features : (offset + 1) * features] = observations[:-age]
    return out


def mirror_stacked_observations(observations):
    """Mirror road geometry left/right for map-independent augmentation."""
    values = np.asarray(observations, dtype=np.float32)
    one = values.ndim == 1
    rows = values.reshape(1, -1).copy() if one else values.copy()
    if rows.shape[1] % NUM_FEATURES:
        raise ValueError("stacked observation width is not a multiple of NUM_FEATURES")
    for frame in range(rows.shape[1] // NUM_FEATURES):
        base = frame * NUM_FEATURES
        for scan_row in range(4):
            left = base + scan_row * 3
            right = left + 1
            rows[:, [left, right]] = rows[:, [right, left]]
        road_left = base + ROAD_LEFT_INDEX
        road_right = base + ROAD_RIGHT_INDEX
        rows[:, [road_left, road_right]] = rows[:, [road_right, road_left]]
        ray_start = base + RAY_START_INDEX
        ray_end = ray_start + NUM_RAY_FEATURES
        rows[:, ray_start:ray_end] = rows[:, ray_start:ray_end][:, ::-1].copy()
    return rows[0] if one else rows


def mirror_action_codes(actions):
    """Swap left/right steering and shoulders in flattened 0..8 actions."""
    actions = np.asarray(actions, dtype=np.int64)
    steer = actions // 3
    shoulder = actions % 3
    steer = np.where(steer == 1, 2, np.where(steer == 2, 1, steer))
    shoulder = np.where(
        shoulder == 1, 2, np.where(shoulder == 2, 1, shoulder)
    )
    return (steer * 3 + shoulder).astype(np.int64)


def upgrade_legacy_stacked_observations(observations, stack):
    """Carry old 15-feature corrections into the ray model neutrally.

    Historical runs did not save raw frames, so their rays cannot be
    reconstructed.  Their proven row geometry and corrected actions remain
    useful: copy those fields and set all unknown rays to the neutral
    midpoint.  New synthetic/live data is what teaches actual ray meaning.
    """
    old = np.asarray(observations, dtype=np.float32)
    if old.ndim != 2 or old.shape[1] != stack * LEGACY_NUM_FEATURES:
        return old
    upgraded = np.zeros((len(old), stack * NUM_FEATURES), dtype=np.float32)
    for frame in range(stack):
        old_base = frame * LEGACY_NUM_FEATURES
        new_base = frame * NUM_FEATURES
        upgraded[:, new_base : new_base + LEGACY_NUM_FEATURES] = old[
            :, old_base : old_base + LEGACY_NUM_FEATURES
        ]
        upgraded[
            :, new_base + RAY_START_INDEX : new_base + RAY_START_INDEX + NUM_RAY_FEATURES
        ] = 0.5
    return upgraded


def fit_balanced_linear(observations, labels):
    """Ridge-fit all nine actions while giving rare corrections a fair vote."""
    x = np.asarray(observations, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    design = np.column_stack((x, np.ones(len(x), dtype=np.float32)))
    targets = np.eye(NUM_ACTIONS, dtype=np.float32)[y]
    counts = np.bincount(y, minlength=NUM_ACTIONS).astype(np.float64)
    present = counts > 0
    sample_weight = len(y) / (present.sum() * counts[y])
    sample_weight = np.minimum(sample_weight, 10.0)
    weighted = design * np.sqrt(sample_weight[:, None])
    weighted_targets = targets * np.sqrt(sample_weight[:, None])
    ridge = np.eye(design.shape[1], dtype=np.float64) * 1e-3
    ridge[-1, -1] = 0.0
    coefficients = np.linalg.solve(
        weighted.T @ weighted + ridge, weighted.T @ weighted_targets
    )
    genome = np.concatenate(
        (coefficients[:-1].astype(np.float32).ravel(),
         coefficients[-1].astype(np.float32))
    )
    agreement = float((np.argmax(design @ coefficients, axis=1) == y).mean())
    return genome, agreement


def lift_linear_genome(linear_genome, inputs, hidden):
    """Embed a trained linear policy in a tanh MLP instead of starting blind.

    Nine hidden units carry the nine linear logits at a small scale. Tanh is
    monotonic, so reading each unit directly preserves the argmax exactly
    without huge output weights that would resist later mutation. Remaining
    hidden units begin dormant and are available for evolution to discover
    nonlinear ray combinations.
    """
    if hidden < NUM_ACTIONS:
        return None
    linear = np.asarray(linear_genome, dtype=np.float32)
    cut = inputs * NUM_ACTIONS
    if linear.size != cut + NUM_ACTIONS:
        raise ValueError("linear seed has the wrong size")
    linear_weights = linear[:cut].reshape(inputs, NUM_ACTIONS)
    linear_bias = linear[cut:]
    scale = 0.05
    w1 = np.zeros((inputs, hidden), dtype=np.float32)
    b1 = np.zeros(hidden, dtype=np.float32)
    w2 = np.zeros((hidden, NUM_ACTIONS), dtype=np.float32)
    b2 = np.zeros(NUM_ACTIONS, dtype=np.float32)
    w1[:, :NUM_ACTIONS] = linear_weights * scale
    b1[:NUM_ACTIONS] = linear_bias * scale
    w2[:NUM_ACTIONS, :] = np.eye(NUM_ACTIONS, dtype=np.float32)
    return np.concatenate((w1.ravel(), b1, w2.ravel(), b2))


def synthetic_wall_samples(stack, count=12000, seed=1729):
    """Varied wall approaches plus exact left/right mirrors.

    These are not Mute-City checkpoint scripts.  They describe the general
    visual rule the live policy needs on any track: a wall that is closer in
    newer frames or farther-ahead scan rows calls for steering and leaning
    toward the open side.  Mirroring makes both directions equally common.
    """
    rng = np.random.default_rng(seed)
    base_count = max(int(count) // 2, 1)
    observations = np.zeros((base_count, stack * NUM_FEATURES), dtype=np.float32)
    labels = np.zeros(base_count, dtype=np.int64)

    for sample in range(base_count):
        safe_straight = sample % 4 == 0
        newest_hazard = rng.uniform(0.05, 0.32)
        closing = rng.uniform(0.20, 0.55)
        curve_strength = rng.uniform(0.04, 0.18)
        for age in range(stack):
            recency = age / max(stack - 1, 1)
            base = age * NUM_FEATURES
            if safe_straight:
                shift = rng.uniform(-0.08, 0.08)
                left_near = np.clip(0.72 + shift, 0.45, 0.95)
                right_near = np.clip(0.72 - shift, 0.45, 0.95)
            else:
                # Build a left-wall approach; the mirrored half below builds
                # the matching right-wall case exactly.
                left_near = np.clip(newest_hazard + closing * (1.0 - recency), 0.0, 1.0)
                right_near = rng.uniform(0.62, 0.95)

            for scan_row in range(4):
                row = base + scan_row * 3
                ahead = 3 - scan_row
                left = left_near if safe_straight else left_near - ahead * curve_strength
                observations[sample, row] = np.clip(left, 0.0, 1.0)
                observations[sample, row + 1] = np.clip(right_near, 0.0, 1.0)
                observations[sample, row + 2] = 1.0
            observations[sample, base + SPEED_INDEX] = rng.uniform(0.22, 0.85)
            if safe_straight:
                rays = np.array((0.55, 0.70, 0.85, 0.95, 0.85, 0.70, 0.55))
                rays += rng.normal(0.0, 0.025, NUM_RAY_FEATURES)
            else:
                # Left-facing rays shorten as the left wall closes; the
                # right-facing fan remains open.  Exact mirroring below
                # creates the opposite turn with no directional bias.
                rays = np.linspace(left_near, right_near, NUM_RAY_FEATURES)
                rays[3] = 0.5 * (left_near + right_near)
                rays += rng.normal(0.0, 0.02, NUM_RAY_FEATURES)
            observations[
                sample, base + RAY_START_INDEX : base + RAY_START_INDEX + NUM_RAY_FEATURES
            ] = np.clip(rays, 0.0, 1.0)

        if not safe_straight:
            steer = 2  # away from the synthetic left-wall approach
            shoulder = 2 if newest_hazard < 0.28 else 0
            labels[sample] = steer * 3 + shoulder

    mirrored = mirror_stacked_observations(observations)
    mirrored_labels = mirror_action_codes(labels)
    return (
        np.concatenate((observations, mirrored)).astype(np.float32),
        np.concatenate((labels, mirrored_labels)).astype(np.int64),
    )


def supervised_wall_warmup(genome, stack, inputs, hidden, steps=250, seed=7331):
    """Backpropagate general seven-ray wall avoidance into a copied MLP.

    Evolution still decides whether this candidate can actually drive.  The
    original inherited elites remain untouched; this creates one additional
    candidate whose hidden units have already learned that asymmetric and
    closing ray fans require the matching steer and shoulder action.
    """
    if hidden <= 0 or steps <= 0:
        return np.asarray(genome, dtype=np.float32).copy(), "warm-up disabled"

    x, y = synthetic_wall_samples(stack, count=8000, seed=seed)
    rng = np.random.default_rng(seed)
    cut = inputs * hidden
    cut2 = cut + hidden
    cut3 = cut2 + hidden * NUM_ACTIONS
    params = [
        np.asarray(genome[:cut], dtype=np.float32).reshape(inputs, hidden).copy(),
        np.asarray(genome[cut:cut2], dtype=np.float32).copy(),
        np.asarray(genome[cut2:cut3], dtype=np.float32).reshape(hidden, NUM_ACTIONS).copy(),
        np.asarray(genome[cut3:], dtype=np.float32).copy(),
    ]

    # Give dormant units a tiny, symmetric starting signal.  A lifted linear
    # teacher otherwise leaves 23/32 units exactly disconnected forever until
    # several coordinated mutations happen to activate both sides of a unit.
    dormant = np.linalg.norm(params[0], axis=0) + np.linalg.norm(params[2], axis=1) < 1e-8
    params[0][:, dormant] = rng.normal(0.0, 0.01, (inputs, int(dormant.sum())))
    params[2][dormant, :] = rng.normal(0.0, 0.01, (int(dormant.sum()), NUM_ACTIONS))

    counts = np.bincount(y, minlength=NUM_ACTIONS).astype(np.float32)
    class_weight = np.zeros(NUM_ACTIONS, dtype=np.float32)
    present = counts > 0
    class_weight[present] = len(y) / (present.sum() * counts[present])
    first_moment = [np.zeros_like(value) for value in params]
    second_moment = [np.zeros_like(value) for value in params]
    beta1, beta2, learning_rate = 0.9, 0.999, 0.004

    for step in range(1, int(steps) + 1):
        chosen = rng.integers(0, len(x), size=min(256, len(x)))
        batch = x[chosen]
        labels = y[chosen]
        w1, b1, w2, b2 = params
        hidden_values = np.tanh(batch @ w1 + b1)
        logits = hidden_values @ w2 + b2
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        weights = class_weight[labels]
        output_delta = probabilities
        output_delta[np.arange(len(labels)), labels] -= 1.0
        output_delta *= (weights / max(weights.sum(), 1e-6))[:, None]
        hidden_delta = (output_delta @ w2.T) * (1.0 - hidden_values * hidden_values)
        gradients = [
            batch.T @ hidden_delta,
            hidden_delta.sum(axis=0),
            hidden_values.T @ output_delta,
            output_delta.sum(axis=0),
        ]
        for index, gradient in enumerate(gradients):
            np.clip(gradient, -1.0, 1.0, out=gradient)
            first_moment[index] = beta1 * first_moment[index] + (1.0 - beta1) * gradient
            second_moment[index] = beta2 * second_moment[index] + (1.0 - beta2) * gradient * gradient
            corrected_first = first_moment[index] / (1.0 - beta1 ** step)
            corrected_second = second_moment[index] / (1.0 - beta2 ** step)
            params[index] -= learning_rate * corrected_first / (np.sqrt(corrected_second) + 1e-8)

    warmed = np.concatenate((params[0].ravel(), params[1], params[2].ravel(), params[3]))
    actions = action_signature(warmed, x, inputs, hidden)
    agreement = float(np.mean(actions == y))
    return warmed.astype(np.float32), f"{steps} hidden-layer BC steps; {agreement * 100:.1f}% wall-pattern agreement"


def emphasise_ray_connections(genome, stack, inputs, hidden, gain=2.0):
    """Return a candidate with stronger seven-ray input connections only."""
    result = np.asarray(genome, dtype=np.float32).copy()
    ray_rows = [
        age * NUM_FEATURES + feature
        for age in range(stack)
        for feature in range(RAY_START_INDEX, RAY_START_INDEX + NUM_RAY_FEATURES)
    ]
    if hidden > 0:
        cut = inputs * hidden
        weights = result[:cut].reshape(inputs, hidden)
        weights[ray_rows, :] *= float(gain)
    else:
        cut = inputs * NUM_ACTIONS
        weights = result[:cut].reshape(inputs, NUM_ACTIONS)
        weights[ray_rows, :] *= float(gain)
    return result


def latest_correction_run():
    for run_id in list_runs():
        if load_meta(run_id).get("source") != "neuroevolution":
            continue
        for episode in reversed(load_episodes(run_id)):
            steps = load_steps(run_id, episode)
            if steps is not None and "applied_actions" in steps:
                return run_id
    return None


def correction_samples(run_id, stack, inputs, max_samples=8000):
    """Aligned observation -> safety-corrected action examples from the hive mind."""
    if run_id == "auto":
        run_id = latest_correction_run()
    if not run_id:
        return None, None, "no prior correction run was found"

    observations = []
    labels = []
    # Recent rules-compatible corrections are more useful than old behavior,
    # and bounding this keeps a restart from loading thousands of compressed
    # episodes before the emulators can begin work.
    for episode in load_episodes(run_id)[-500:]:
        steps = load_steps(run_id, episode)
        if steps is None or len(steps.get("observations", ())) <= stack:
            continue
        applied = steps.get("applied_actions")
        overridden = steps.get("safety_override")
        if applied is None or overridden is None or applied.shape[1] != 2:
            continue

        # Writer rows contain the observation *after* their action.  Action
        # t+1 therefore aligns with the stack ending at observation t.
        stacked = stack_episode(steps["observations"], stack)[:-1]
        applied = applied[1:]
        keep = overridden[1:].astype(bool)
        keep &= steps.get("racing", np.ones(len(keep) + 1, dtype=bool))[1:]
        keep &= ~steps.get("wall", np.zeros(len(keep) + 1, dtype=bool))[1:]
        if keep.any():
            observations.append(stacked[keep])
            labels.append((applied[keep, 0] * 3 + applied[keep, 1]).astype(np.int64))

    if not observations:
        return None, None, f"run '{run_id}' has no clean safety corrections"
    x = np.concatenate(observations)
    y = np.concatenate(labels)
    x = upgrade_legacy_stacked_observations(x, stack)
    if x.shape[1] != inputs:
        return None, None, f"corrections have {x.shape[1]} inputs, policy expects {inputs}"
    if len(x) > max_samples:
        keep = np.linspace(0, len(x) - 1, max_samples, dtype=np.int64)
        x, y = x[keep], y[keep]

    # Exact mirror augmentation prevents a one-direction track from teaching
    # a one-direction policy, which is essential for transfer to new maps.
    x = np.concatenate((x, mirror_stacked_observations(x)))
    y = np.concatenate((y, mirror_action_codes(y)))
    return x, y, f"{run_id}: {len(y) // 2} clean corrections plus mirrors"


def fit_guidance_seed(run_id, stack, inputs, hidden):
    synthetic_x, synthetic_y = synthetic_wall_samples(stack)
    correction_x, correction_y, note = correction_samples(run_id, stack, inputs)
    if correction_x is not None:
        x = np.concatenate((synthetic_x, correction_x))
        y = np.concatenate((synthetic_y, correction_y))
    else:
        x, y = synthetic_x, synthetic_y
    genome, agreement = fit_balanced_linear(x, y)
    if hidden > 0:
        genome = lift_linear_genome(genome, inputs, hidden)
        if genome is None:
            return None, f"hidden layer needs at least {NUM_ACTIONS} units for guidance"
    return genome, f"{note}; {len(synthetic_y)} varied wall samples; {agreement * 100:.1f}% fit"


def latest_teacher_run():
    for run_id in list_runs():
        if load_meta(run_id).get("source") == "autopilot" and load_episodes(run_id):
            return run_id
    return None


def fit_teacher_seed(run_id, stack, inputs, hidden, max_samples=50000):
    """Fit a balanced linear steer+lean policy to saved autopilot actions."""
    if run_id == "auto":
        run_id = latest_teacher_run()
    if not run_id:
        return None, "no saved autopilot run was found"

    observations = []
    labels = []
    for episode in load_episodes(run_id):
        steps = load_steps(run_id, episode)
        if steps is None or len(steps["observations"]) < stack:
            continue
        # Stored observations are post-action.  Shift labels one row so each
        # action is paired with the stack the controller actually saw.
        obs = stack_episode(steps["observations"], stack)[:-1]
        actions = steps["actions"]
        if actions.ndim != 2 or actions.shape[1] not in (2, 3):
            continue
        actions = actions[1:]
        steer = actions[:, 0]
        shoulder = actions[:, 1] if actions.shape[1] == 2 else actions[:, 2]
        live = steps.get("racing", np.ones(len(obs) + 1, dtype=bool))[1:]
        clean = ~steps.get("wall", np.zeros(len(obs) + 1, dtype=bool))[1:]
        keep = live & clean
        observations.append(obs[keep])
        labels.append((steer[keep] * 3 + shoulder[keep]).astype(np.int64))

    if not observations:
        return None, f"teacher run '{run_id}' has no usable step arrays"
    x = np.concatenate(observations)
    y = np.concatenate(labels)
    x = upgrade_legacy_stacked_observations(x, stack)
    if len(x) > max_samples:
        # Deterministic spacing keeps the seed repeatable while sampling the
        # entire collection instead of taking only its earliest episodes.
        keep = np.linspace(0, len(x) - 1, max_samples, dtype=np.int64)
        x, y = x[keep], y[keep]
    if x.shape[1] != inputs:
        return None, f"teacher has {x.shape[1]} inputs, policy expects {inputs}"

    genome, agreement = fit_balanced_linear(x, y)
    if hidden > 0:
        genome = lift_linear_genome(genome, inputs, hidden)
        if genome is None:
            return None, f"hidden layer needs at least {NUM_ACTIONS} units for teacher seed"
    return genome, f"{run_id}: {len(x)} clean steps, {agreement * 100:.1f}% agreement"


def analytic_teacher_seed(inputs, stack):
    """Controller-shaped fallback when no recorded teacher data is available."""
    weights = np.zeros((inputs, NUM_ACTIONS), dtype=np.float32)
    bias = np.full(NUM_ACTIONS, -0.25, dtype=np.float32)
    bias[0] = 0.10  # straight, no lean
    row_strength = (0.45, 0.60, 0.80, 1.00)
    for age in range(stack):
        age_strength = (age + 1) / stack
        base = age * NUM_FEATURES
        for row, strength in enumerate(row_strength):
            left = base + row * 3
            right = left + 1
            scale = strength * age_strength
            # More room on the right means the car is near the left wall.
            weights[right, 6] += scale
            weights[left, 6] -= scale
            weights[left, 3] += scale
            weights[right, 3] -= scale
            weights[right, 8] += 0.75 * scale
            weights[left, 8] -= 0.75 * scale
            weights[left, 4] += 0.75 * scale
            weights[right, 4] -= 0.75 * scale
    bias[4] = bias[8] = -0.55  # lean only for a strong correction
    return np.concatenate((weights.ravel(), bias))


def initial_population(size, population_size, inputs, hidden, stack, rng,
                       teacher_run="auto", seed_model=None, allow_teacher=True):
    """Seed a fresh population, optionally from a demonstration teacher.

    allow_teacher=False means NO demonstration seeding of any kind. That is
    not the same as teacher_run=None, which used to fall through to
    analytic_teacher_seed -- and the analytic seed is the hand-written
    controller, which drives straight until it meets a wall. Seeding six
    genomes with that, then evolving from them, is one of the ways a
    straight-line lineage kept re-entering the population even when the
    autopilot run was explicitly excluded.
    """
    population = []
    seed = None
    note = "no demonstration seed (teacher and analytic fallback both disabled)"

    if allow_teacher:
        seed, note = fit_teacher_seed(teacher_run, stack, inputs, hidden)
        if seed is None:
            seed = analytic_teacher_seed(inputs, stack)
            if hidden > 0:
                seed = lift_linear_genome(seed, inputs, hidden)
            note = f"analytic controller fallback ({note})"

    if seed is not None:
        population.append(seed.astype(np.float32))
        seeded = max(1, min(population_size // 4, 6))
        for _ in range(1, seeded):
            population.append(
                (seed + rng.normal(0.0, 0.025, size)).astype(np.float32)
            )

    if seed_model and os.path.exists(seed_model):
        with np.load(seed_model) as model:
            prior = model["weights"].astype(np.float32)
        if prior.size == size:
            population.insert(0, prior)
            note += f"; included {seed_model}"

    population += [
        random_immigrant(rng, size, inputs=inputs, hidden=hidden)
        for _ in range(max(population_size - len(population), 0))
    ]
    return population[:population_size], note


def genome_size(inputs, hidden):
    if hidden <= 0:
        return inputs * NUM_ACTIONS + NUM_ACTIONS
    return inputs * hidden + hidden + hidden * NUM_ACTIONS + NUM_ACTIONS


def act(genome, observation, inputs, hidden):
    """Greedy action from the policy encoded in `genome`."""
    if hidden <= 0:
        weights = genome[: inputs * NUM_ACTIONS].reshape(inputs, NUM_ACTIONS)
        bias = genome[inputs * NUM_ACTIONS :]
        return int(np.argmax(observation @ weights + bias))

    cut = inputs * hidden
    w1 = genome[:cut].reshape(inputs, hidden)
    b1 = genome[cut : cut + hidden]
    cut2 = cut + hidden
    w2 = genome[cut2 : cut2 + hidden * NUM_ACTIONS].reshape(hidden, NUM_ACTIONS)
    b2 = genome[cut2 + hidden * NUM_ACTIONS :]
    # tanh, not relu: bounded, so a mutation cannot blow one unit up and
    # swamp the rest of the layer.
    return int(np.argmax(np.tanh(observation @ w1 + b1) @ w2 + b2))


def action_signature(genome, observations, inputs, hidden):
    """Actions on fixed synthetic situations, used only to preserve diversity."""
    observations = np.asarray(observations, dtype=np.float32)
    if hidden <= 0:
        weights = genome[: inputs * NUM_ACTIONS].reshape(inputs, NUM_ACTIONS)
        bias = genome[inputs * NUM_ACTIONS :]
        return np.argmax(observations @ weights + bias, axis=1).astype(np.int8)
    cut = inputs * hidden
    w1 = genome[:cut].reshape(inputs, hidden)
    b1 = genome[cut : cut + hidden]
    cut2 = cut + hidden
    w2 = genome[cut2 : cut2 + hidden * NUM_ACTIONS].reshape(hidden, NUM_ACTIONS)
    b2 = genome[cut2 + hidden * NUM_ACTIONS :]
    return np.argmax(np.tanh(observations @ w1 + b1) @ w2 + b2, axis=1).astype(np.int8)


def diverse_parent_indices(population, scores, probes, inputs, hidden, count):
    """Pick good parents that react differently on mirrored wall/turn probes.

    Fitness limits eligibility to the requested good-parent pool, so novelty
    cannot promote a plainly worse checkpoint result.  Inside that set, greedy Hamming
    distance prevents every parent from collapsing to the same turn script.
    """
    count = min(max(int(count), 1), len(population))
    order = sorted(range(len(population)), key=lambda i: rank_key(scores[i]), reverse=True)
    eligible = order[:count]
    signatures = {
        index: action_signature(population[index], probes, inputs, hidden)
        for index in eligible
    }
    selected = [eligible[0]]
    while len(selected) < count:
        remaining = [index for index in eligible if index not in selected]
        if not remaining:
            break
        rank_position = {index: position for position, index in enumerate(order)}

        def choice_key(index):
            novelty = min(
                float(np.mean(signatures[index] != signatures[other]))
                for other in selected
            )
            return novelty, -rank_position[index]

        selected.append(max(remaining, key=choice_key))
    return selected


def run_episode(environment, recorder, stacker, genome, inputs, hidden,
                start_slot=None, start_steers=()):
    reset_options = {"start_steers": tuple(start_steers)}
    if start_slot is not None:
        reset_options["start_slot"] = start_slot
    observation, info = recorder.reset(options=reset_options)
    stacked = stacker.reset(observation)
    wall_steps = 0
    centre_total = 0.0
    speed_total = 0.0
    risk_total = 0.0
    steps = 0

    while True:
        action = act(genome, stacked, inputs, hidden)
        # Flattened 0..8 for the genome; the environment receives its two
        # independent controls: steer (0..2), then lean (0..2).
        observation, _, terminated, truncated, info = recorder.step(
            (action // 3, action % 3)
        )
        stacked = stacker.push(observation)
        steps += 1
        wall_steps += bool(info.get("wall"))
        centre_total += 1.0 - min(max(float(info.get("off_centre", 1.0)), 0.0), 1.0)
        speed_total += min(max(float(info.get("speed", 0)) / TOP_SPEED_RAW, 0.0), 1.0)
        risk_total += min(max(float(info.get("wall_risk", 0.0)), 0.0), 1.0)
        if terminated or truncated:
            break

    summary = recorder.last_summary or {}
    # "furthest" is now DISTANCE DRIVEN from the spawn, already relative, so
    # the fallback must not subtract start_progress a second time. It used to
    # be an absolute checkpoint number, which is why that subtraction existed.
    absolute_furthest = summary.get("furthest", info.get("furthest", 0))
    progress_gain = summary.get(
        "progress_gain",
        info.get("progress_gain", max(absolute_furthest, 0)),
    )
    signed_distance = summary.get("net_progress", info.get("net_progress", progress_gain))
    return {
        # Mixed turn slots must be compared by distance driven from their
        # own start, not by the checkpoint number they were handed for free.
        "furthest": progress_gain,
        "distance": signed_distance,
        "absolute_furthest": absolute_furthest,
        "start_slot": summary.get("start_slot", info.get("start_slot")),
        "start_variant": summary.get(
            "start_variant", info.get("start_variant", list(start_steers))
        ),
        "wall_rate": summary.get("wall_rate", wall_steps / max(steps, 1)),
        "centre": summary.get(
            "centre_quality",
            1.0 - float(
                info.get("mean_off_centre", 1.0 - centre_total / max(steps, 1))
            ),
        ),
        "speed": float(
            summary.get("mean_speed", 0.0) / TOP_SPEED_RAW
            if summary
            else info.get("mean_speed_normalised", speed_total / max(steps, 1))
        ),
        "speed_loss": float(
            summary.get(
                "mean_speed_loss",
                info.get("mean_speed_loss", 0.0),
            )
        ),
        "ray_response": float(
            summary.get(
                "ray_response_score",
                info.get("ray_response_score", 0.0),
            )
        ),
        "ray_clearance_recovery": float(
            summary.get(
                "mean_ray_clearance_recovery",
                info.get("mean_ray_clearance_recovery", 0.0),
            )
        ),
        "wall_risk": summary.get(
            "mean_wall_risk",
            info.get("mean_wall_risk", risk_total / max(steps, 1)),
        ),
        "safety_override_rate": summary.get(
            "safety_override_rate", info.get("safety_override_rate", 0.0)
        ),
        "policy_agreement": summary.get("policy_agreement", 0.0),
        "actual_lean_rate": summary.get(
            "actual_lean_rate", info.get("actual_lean_rate", 0.0)
        ),
        "frontier_clearance": summary.get("frontier_clearance", 0.0),
        "frontier_centre": summary.get("frontier_centre", 0.0),
        "turn_recovery": summary.get("turn_recovery", 0.0),
        "safe_frontier_rate": summary.get("safe_frontier_rate", 0.0),
        "turn_quality": summary.get("turn_quality", 0.0),
        "curriculum": int(bool(info.get("curriculum_complete"))),
        "lap": int(bool(info.get("finished"))),
    }


def worker(context, jobs, episodes):
    (
        instance,
        run_id,
        frame_skip,
        stack,
        inputs,
        hidden,
        target_progress,
        start_slots,
        evaluation_frames,
        generation,
        end_on_wall,
        wall_contact_frames,
    ) = context
    environment = FZeroEnv(
        instance=instance,
        frame_skip=frame_skip,
        curriculum_checkpoints=0 if evaluation_frames else target_progress,
        assist_mode="emergency",
        start_slots=start_slots,
        episode_frame_limit=evaluation_frames or MAX_EPISODE_FRAMES,
        # A wall collision consumes time and speed in a time trial, but does
        # not erase the chance to demonstrate recovery. Destruction and ten
        # seconds without progress still end hopeless runs early.
        terminate_on_wall=bool(end_on_wall) or not bool(evaluation_frames),
        terminate_on_stall=not bool(evaluation_frames),
        wall_contact_frames=wall_contact_frames,
    )
    writer = RunWriter(
        run_id,
        instance=instance,
        # Store the human-facing 1-based number shown by the GA console.
        # This is exact even when staged screening creates two worker batches
        # for one generation.
        episode_fields={"generation": generation + 1},
        meta={
            "source": "neuroevolution",
            "hidden": hidden,
            "inputs": inputs,
            "curriculum_checkpoints": target_progress,
            "start_slots": list(start_slots),
            "assist_mode": "emergency",
            "evolution_rules_version": EVOLUTION_RULES_VERSION,
            "evaluation_frames": evaluation_frames,
            "end_on_wall": bool(end_on_wall),
            "wall_contact_frames": wall_contact_frames,
        },
    )
    recorder = RecordingWrapper(environment, writer)
    stacker = Stacker(stack)

    try:
        while True:
            try:
                genome_index, repeat_index, genome, start_slot, start_steers = (
                    jobs.get_nowait()
                )
            except queue.Empty:
                return

            try:
                recorder.writer.episode_fields.update({
                    "genome_id": genome_id(genome),
                    "evaluation_repeat": int(repeat_index),
                })
                episodes[genome_index][repeat_index] = run_episode(
                    environment, recorder, stacker, genome, inputs, hidden,
                    start_slot, start_steers,
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


def evaluation_scenarios(start_slots, repeats, repeat_offset=0, generation=0):
    """Return the shared deterministic conditions for one evaluation batch.

    A scenario is a savestate plus a short steering prelude. All genomes see
    the same scenario for a given repeat, while ``generation`` rotates both
    parts. Staged screening can split a batch and recover the exact same
    schedule by passing ``repeat_offset`` for its promoted extra episodes.
    """
    scenarios = []
    slot_count = len(start_slots)
    for local_repeat in range(repeats):
        repeat_index = repeat_offset + local_repeat
        slot = (
            start_slots[(repeat_index + generation) % slot_count]
            if start_slots else None
        )
        variant_index = (repeat_index + generation) % len(START_VARIANTS)
        scenarios.append((slot, variant_index, START_VARIANTS[variant_index]))
    return scenarios


def evaluate_population(population, instances, run_id, repeats, frame_skip, stack,
                        inputs, hidden, target_progress=0, start_slots=(),
                        repeat_offset=0, instance_offset=0, generation=0,
                        evaluation_frames=None, end_on_wall=False,
                        wall_contact_frames=WALL_CONTACT_FRAMES_LIMIT):
    jobs = queue.Queue()
    # Interleaved by repeat, so a dead emulator costs every genome its last
    # episode rather than costing the late ones all of theirs.
    #
    # A mere permutation of slots 2,3,4 would add no variety when all three
    # are evaluated every generation. Pair them with small real start-state
    # perturbations instead.
    scenarios = evaluation_scenarios(
        start_slots, repeats, repeat_offset=repeat_offset, generation=generation
    )
    for repeat_index, (start_slot, _variant_index, start_steers) in enumerate(scenarios):
        for genome_index, genome in enumerate(population):
            jobs.put((genome_index, repeat_index, genome, start_slot, start_steers))

    episodes = [[None] * repeats for _ in population]
    threads = []
    for instance in range(instance_offset, instance_offset + instances):
        thread = threading.Thread(
            target=worker,
            args=(
                (
                    instance,
                    run_id,
                    frame_skip,
                    stack,
                    inputs,
                    hidden,
                    target_progress,
                    start_slots,
                    evaluation_frames,
                    generation,
                    end_on_wall,
                    wall_contact_frames,
                ),
                jobs,
                episodes,
            ),
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    return [
        [episode or dict(DEAD_EPISODE) for episode in genome_episodes]
        for genome_episodes in episodes
    ]


def score_population(population, instances, run_id, repeats, frame_skip, stack,
                     inputs, hidden, history=None, target_progress=0, start_slots=(),
                     instance_offset=0, generation=0, evaluation_frames=None,
                     end_on_wall=False,
                     wall_contact_frames=WALL_CONTACT_FRAMES_LIMIT):
    episode_sets = evaluate_population(
        population, instances, run_id, repeats, frame_skip, stack, inputs,
        hidden, target_progress, start_slots, instance_offset=instance_offset,
        generation=generation, evaluation_frames=evaluation_frames,
        end_on_wall=end_on_wall, wall_contact_frames=wall_contact_frames,
    )
    scores = []
    for genome, fresh in zip(population, episode_sets):
        pooled = history.add(genome, fresh) if history is not None else fresh
        scores.append(summarise(pooled))
    return add_ray_probe_fitness(population, scores, stack, inputs, hidden)


def score_population_staged(population, instances, run_id, repeats, frame_skip,
                            stack, inputs, hidden, history, target_progress,
                            screen_repeats, screen_fraction, start_slots=(),
                            instance_offset=0, generation=0,
                            evaluation_frames=None, end_on_wall=False,
                            wall_contact_frames=WALL_CONTACT_FRAMES_LIMIT):
    """Give all genomes two runs, then spend remaining runs on the best 75%."""
    screen_repeats = min(max(int(screen_repeats), 1), repeats)
    if screen_repeats >= repeats or screen_fraction >= 1.0:
        return (
            score_population(
                population, instances, run_id, repeats, frame_skip, stack,
                inputs, hidden, history, target_progress, start_slots,
                instance_offset=instance_offset, generation=generation,
                evaluation_frames=evaluation_frames,
                end_on_wall=end_on_wall,
                wall_contact_frames=wall_contact_frames,
            ),
            len(population),
        )

    episode_sets = evaluate_population(
        population, instances, run_id, screen_repeats, frame_skip, stack,
        inputs, hidden, target_progress, start_slots,
        instance_offset=instance_offset, generation=generation,
        evaluation_frames=evaluation_frames,
        end_on_wall=end_on_wall, wall_contact_frames=wall_contact_frames,
    )
    screened = add_ray_probe_fitness(
        population,
        [summarise(episodes) for episodes in episode_sets],
        stack,
        inputs,
        hidden,
    )
    promoted_count = max(
        ELITES,
        min(len(population), int(np.ceil(len(population) * screen_fraction))),
    )
    promoted = sorted(
        range(len(population)), key=lambda i: rank_key(screened[i]), reverse=True
    )[:promoted_count]
    promoted_population = [population[i] for i in promoted]
    extra_sets = evaluate_population(
        promoted_population, instances, run_id, repeats - screen_repeats,
        frame_skip, stack, inputs, hidden, target_progress, start_slots,
        repeat_offset=screen_repeats, instance_offset=instance_offset,
        generation=generation,
        evaluation_frames=evaluation_frames,
        end_on_wall=end_on_wall, wall_contact_frames=wall_contact_frames,
    )
    for index, extra in zip(promoted, extra_sets):
        episode_sets[index].extend(extra)

    scores = []
    for genome, fresh in zip(population, episode_sets):
        pooled = history.add(genome, fresh)
        scores.append(summarise(pooled))
    return add_ray_probe_fitness(population, scores, stack, inputs, hidden), promoted_count


class WeightHistory(History):
    """History keyed on weight vectors rather than parameter dicts."""

    @staticmethod
    def key(genome):
        return genome.tobytes()


def breed(parent_a, parent_b, rng, size, inputs=None, hidden=0, sigma=MUTATION_SIGMA):
    """Coherent action crossover followed by weight and bias mutation.

    The old per-weight crossover shredded useful action columns.  For the
    default linear policy, each child now inherits a complete action column
    from one parent, then explores inside it.  Biases receive stronger noise
    because a dominant bias was what collapsed the live model to one action.
    """
    if hidden <= 0 and inputs and size == inputs * NUM_ACTIONS + NUM_ACTIONS:
        cut = inputs * NUM_ACTIONS
        weights_a = parent_a[:cut].reshape(inputs, NUM_ACTIONS)
        weights_b = parent_b[:cut].reshape(inputs, NUM_ACTIONS)
        bias_a = parent_a[cut:]
        bias_b = parent_b[cut:]
        take_a = rng.random(NUM_ACTIONS) < 0.5
        weights = np.where(take_a[None, :], weights_a, weights_b).copy()
        bias = np.where(take_a, bias_a, bias_b).copy()

        ray_rows = ray_input_mask(inputs)
        rates = np.full(weights.shape, MUTATION_RATE, dtype=np.float32)
        scales = np.ones(weights.shape, dtype=np.float32)
        rates[ray_rows, :] = min(MUTATION_RATE * RAY_MUTATION_RATE_GAIN, 1.0)
        scales[ray_rows, :] = RAY_MUTATION_SIGMA_GAIN
        mutate = rng.random(weights.shape) < rates
        weights += mutate * rng.normal(0.0, sigma, weights.shape) * scales
        bias_mutate = rng.random(NUM_ACTIONS) < max(MUTATION_RATE, 0.35)
        bias += bias_mutate * rng.normal(
            0.0, sigma * OUTPUT_BIAS_SIGMA_SCALE, NUM_ACTIONS
        )
        return np.concatenate((weights.ravel(), bias)).astype(np.float32)

    if hidden > 0 and inputs and size == genome_size(inputs, hidden):
        cut = inputs * hidden
        cut2 = cut + hidden
        cut3 = cut2 + hidden * NUM_ACTIONS
        a_w1 = parent_a[:cut].reshape(inputs, hidden)
        b_w1 = parent_b[:cut].reshape(inputs, hidden)
        a_b1, b_b1 = parent_a[cut:cut2], parent_b[cut:cut2]
        a_w2 = parent_a[cut2:cut3].reshape(hidden, NUM_ACTIONS)
        b_w2 = parent_b[cut2:cut3].reshape(hidden, NUM_ACTIONS)
        a_b2, b_b2 = parent_a[cut3:], parent_b[cut3:]

        # A hidden unit is a coherent feature: its incoming column, bias and
        # outgoing row must come from the same parent. Per-weight crossover
        # would destroy that feature before mutation gets a chance to refine it.
        take_a_unit = rng.random(hidden) < 0.5
        w1 = np.where(take_a_unit[None, :], a_w1, b_w1).copy()
        b1 = np.where(take_a_unit, a_b1, b_b1).copy()
        w2 = np.where(take_a_unit[:, None], a_w2, b_w2).copy()
        take_a_output = rng.random(NUM_ACTIONS) < 0.5
        b2 = np.where(take_a_output, a_b2, b_b2).copy()

        rate = min(MUTATION_RATE, 0.12)
        ray_rows = ray_input_mask(inputs)
        rates = np.full(w1.shape, rate, dtype=np.float32)
        scales = np.ones(w1.shape, dtype=np.float32)
        rates[ray_rows, :] = min(rate * RAY_MUTATION_RATE_GAIN, 1.0)
        scales[ray_rows, :] = RAY_MUTATION_SIGMA_GAIN
        w1 += (rng.random(w1.shape) < rates) * rng.normal(0.0, sigma, w1.shape) * scales
        w2 += (rng.random(w2.shape) < rate) * rng.normal(0.0, sigma, w2.shape)
        b1 += (rng.random(b1.shape) < 0.25) * rng.normal(
            0.0, sigma * HIDDEN_BIAS_SIGMA_SCALE, b1.shape
        )
        b2 += (rng.random(b2.shape) < 0.35) * rng.normal(
            0.0, sigma * OUTPUT_BIAS_SIGMA_SCALE, b2.shape
        )
        return np.concatenate((w1.ravel(), b1, w2.ravel(), b2)).astype(np.float32)

    take_a = rng.random(size) < 0.5
    child = np.where(take_a, parent_a, parent_b)
    mutate = rng.random(size) < MUTATION_RATE
    child = child + mutate * rng.normal(0.0, sigma, size)
    return child.astype(np.float32)


def build_next_population(
    ranked,
    population,
    scores,
    picker,
    rng,
    size,
    inputs,
    hidden,
    guidance=None,
    stalled=False,
    diversity_probes=None,
    elite_random_split=False,
    lineage_log=None,
):
    """Targeted refinement, coherent broad mutations and random immigrants.

    Two fitness elites and two behaviorally distinct good genomes survive.
    A correction-guided seed is reintroduced every generation.  Ten percent
    are direct variants of proven elites, about 8% are targeted exploratory
    crossovers, and 20% are random immigrants.  The remaining slots are
    careful small-mutation children of the best 16 behaviorally diverse
    parents.

    After a five-generation plateau a PARTIAL MASS EXTINCTION runs instead of
    the old widened-mutation tier: the random-immigrant share grows to
    EXTINCTION_FRACTION, wiping the weakest lineages and reseeding them from
    scratch, while elites and diversity picks survive untouched and keep
    their normal mutation step.
    """
    target = len(population)
    if lineage_log is not None:
        lineage_log.clear()

    def record(candidate, source, parents=()):
        if lineage_log is not None:
            lineage_log.append({
                "genome_id": genome_id(candidate),
                "source": source,
                "parent_ids": [genome_id(parent) for parent in parents],
            })

    if elite_random_split:
        # Keep half the generation exploratory, but make the learned half
        # actually EVOLVE. The previous implementation copied all top 50%
        # unchanged, so useful ray, steering and lean features living in
        # different winners could never be combined. Preserve only the true
        # champions; fill the rest of the learned half with coherent
        # hidden-unit crossover and mutation from the top-half parent pool.
        random_count = target // 2
        learned_count = target - random_count
        exact_elites = min(ELITES, learned_count, len(ranked))
        survivors = [genome.copy() for genome, _ in ranked[:exact_elites]]
        for survivor in survivors:
            record(survivor, "elite", (survivor,))
        parent_pairs = ranked[: max(exact_elites, min(learned_count, len(ranked)))]
        parent_population = [genome for genome, _ in parent_pairs]
        parent_scores = [score for _, score in parent_pairs]

        child_index = 0
        while len(survivors) < learned_count:
            # One broad child in four can combine substantially different
            # traits; the others refine proven connections without destroying
            # them. breed() keeps each hidden unit coherent during crossover.
            sigma = LARGE_MUTATION_SIGMA if child_index % 4 == 3 else MUTATION_SIGMA
            parent_a = tournament(
                parent_population, parent_scores, picker, TOURNAMENT_SIZE
            )
            parent_b = tournament(
                parent_population, parent_scores, picker, TOURNAMENT_SIZE
            )
            child = breed(
                parent_a, parent_b, rng, size, inputs, hidden, sigma
            )
            survivors.append(child)
            record(
                child,
                "broad_bred" if sigma == LARGE_MUTATION_SIGMA else "bred",
                (parent_a, parent_b),
            )
            child_index += 1

        elite_genome = ranked[0][0] if ranked else None
        turn_bias = turning_bias_for_immigrants(elite_genome)
        for _ in range(random_count):
            immigrant = random_immigrant(
                rng, size, turn_bias, inputs=inputs, hidden=hidden
            )
            survivors.append(immigrant)
            record(immigrant, "random")
        return survivors[:target]

    survivors = [genome.copy() for genome, _ in ranked[:ELITES]]
    for survivor in survivors:
        record(survivor, "elite", (survivor,))

    parent_pairs = ranked[: min(PARENT_POOL_SIZE, len(ranked))]
    parent_population = [genome for genome, _ in parent_pairs]
    parent_scores = [score for _, score in parent_pairs]
    if diversity_probes is not None and len(population) > ELITES:
        parent_indices = diverse_parent_indices(
            population,
            scores,
            diversity_probes,
            inputs,
            hidden,
            min(PARENT_POOL_SIZE, len(population)),
        )
        parent_population = [population[index] for index in parent_indices]
        parent_scores = [scores[index] for index in parent_indices]
        diversity_kept = 0
        for index in parent_indices[1:]:
            candidate = population[index]
            if not any(np.array_equal(candidate, kept) for kept in survivors):
                survivor = candidate.copy()
                survivors.append(survivor)
                record(survivor, "diversity_elite", (candidate,))
                diversity_kept += 1
                if diversity_kept >= DIVERSITY_ELITES:
                    break
    if guidance is not None and len(survivors) < target:
        guided = guidance.copy()
        survivors.append(guided)
        record(guided, "guidance")

    # On a plateau, replace the weakest share of the population outright
    # rather than mutating everyone harder. `survivors` above already holds
    # the fitness elites and the behavioural-diversity picks, so those are
    # never touched -- the extinction only enlarges the random-immigrant
    # slice at the expense of bred children, which are the genomes that were
    # re-searching the same basin.
    immigrant_fraction = (
        max(RANDOM_IMMIGRANT_FRACTION, EXTINCTION_FRACTION)
        if stalled
        else RANDOM_IMMIGRANT_FRACTION
    )
    immigrant_count = max(1, int(round(target * immigrant_fraction)))
    immigrant_count = min(immigrant_count, max(target - len(survivors), 0))

    large_count = max(1, int(round(target * LARGE_MUTANT_FRACTION)))
    elite_variant_count = max(1, int(round(target * ELITE_VARIANT_FRACTION)))
    child_slots = max(
        target - len(survivors) - immigrant_count - elite_variant_count, 0
    )
    large_count = min(large_count, child_slots)
    small_count = child_slots - large_count

    # Sigmas no longer widen on a stall. The extinction above is the response
    # instead, so the surviving lineages keep being refined at their normal
    # step size rather than being shaken apart at the same time their
    # neighbours are replaced.
    small_sigma = MUTATION_SIGMA
    large_sigma = LARGE_MUTATION_SIGMA

    def append_child(sigma, source):
        parent_a = tournament(
            parent_population, parent_scores, picker, TOURNAMENT_SIZE
        )
        parent_b = tournament(
            parent_population, parent_scores, picker, TOURNAMENT_SIZE
        )
        candidate = breed(
            parent_a, parent_b, rng, size, inputs, hidden, sigma
        )
        survivors.append(candidate)
        record(candidate, source, (parent_a, parent_b))

    for _ in range(small_count):
        append_child(small_sigma, "bred")
    for _ in range(large_count):
        append_child(large_sigma, "targeted_bred")
    elite_pool = [genome for genome, _ in ranked[: max(ELITES, 4)]]
    for _ in range(min(elite_variant_count, target - len(survivors))):
        elite = elite_pool[int(rng.integers(0, len(elite_pool)))]
        candidate = breed(
            elite, elite, rng, size, inputs, hidden, small_sigma
        )
        survivors.append(candidate)
        record(candidate, "elite_variant", (elite,))
    # Immigrants are tilted toward turning, sized against the current elite's
    # own straight-vs-turn bias -- see turning_bias_for_immigrants.
    elite_genome = ranked[0][0] if ranked else None
    turn_bias = turning_bias_for_immigrants(elite_genome)
    while len(survivors) < target:
        immigrant = random_immigrant(
            rng, size, turn_bias, inputs=inputs, hidden=hidden
        )
        survivors.append(immigrant)
        record(immigrant, "random")
    result = survivors[:target]
    return result


def diversify_forked_population(population, guidance, rng, size, inputs, hidden,
                                supervised_seed=None, ray_seed=None):
    """Preserve the two old elites but reopen a collapsed fork immediately."""
    out = [genome.copy() for genome in population]
    if guidance is not None and len(out) > ELITES:
        out[ELITES] = guidance.copy()
        for index in range(ELITES + 1, min(ELITES + 1 + GUIDANCE_VARIANTS, len(out))):
            out[index] = breed(
                guidance, guidance, rng, size, inputs, hidden, LARGE_MUTATION_SIGMA
            )
    if supervised_seed is not None and len(out) > ELITES + GUIDANCE_VARIANTS + 1:
        out[ELITES + GUIDANCE_VARIANTS + 1] = supervised_seed.copy()
    if ray_seed is not None and len(out) > ELITES + GUIDANCE_VARIANTS + 2:
        out[ELITES + GUIDANCE_VARIANTS + 2] = ray_seed.copy()
    immigrants = max(1, int(round(len(out) * RANDOM_IMMIGRANT_FRACTION)))
    turn_bias = turning_bias_for_immigrants(out[0] if out else None)
    for index in range(len(out) - immigrants, len(out)):
        out[index] = random_immigrant(rng, size, turn_bias, inputs=inputs, hidden=hidden)
    return out


def update_hall_of_fame(hall, ranked, limit=HALL_OF_FAME_SIZE):
    """Keep strong candidates from every generation, not only the last one."""
    candidates = list(hall or []) + [
        (genome.copy(), dict(score))
        for genome, score in ranked[:HALL_OF_FAME_PER_GENERATION]
    ]
    candidates.sort(key=lambda pair: rank_key(pair[1]), reverse=True)
    selected = []
    seen = set()
    for genome, score in candidates:
        key = genome.tobytes()
        if key in seen:
            continue
        seen.add(key)
        selected.append((genome.copy(), dict(score)))
        if len(selected) >= limit:
            break
    return selected


def save_checkpoint(path, **state):
    """Atomically save a complete GA generation so Ctrl+C is recoverable."""
    state["version"] = CHECKPOINT_VERSION
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_checkpoint(path, expected):
    with open(path, "rb") as handle:
        state = pickle.load(handle)
    if state.get("version") != CHECKPOINT_VERSION:
        raise SystemExit(f"Unsupported GA checkpoint version in {path}")
    for key, value in expected.items():
        if state.get("config", {}).get(key) != value:
            raise SystemExit(
                f"Cannot resume {path}: {key} was "
                f"{state.get('config', {}).get(key)!r}, now {value!r}"
            )
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0,
                        help="first instance number to use -- set this to "
                             "run alongside another trainer on a disjoint "
                             "range, e.g. this on 8+ while DQN uses 0-7")
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                        help="episodes per genome -- do not set this to 1")
    parser.add_argument("--final-repeats", type=int, default=DEFAULT_FINAL_REPEATS)
    parser.add_argument("--frame-skip", type=int, default=DEFAULT_FRAME_SKIP)
    parser.add_argument("--frame-stack", type=int, default=DEFAULT_FRAME_STACK)
    parser.add_argument(
        "--evaluation-seconds",
        type=float,
        default=DEFAULT_EVALUATION_SECONDS,
        help="fixed emulated-time budget per candidate; 0 restores legacy "
             "wall/curriculum termination (default: 30)",
    )
    parser.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN,
                        help=f"hidden units; default {DEFAULT_HIDDEN}, 0 selects legacy linear")
    parser.add_argument(
        "--start-slots",
        type=parse_start_slots,
        default=(2,),
        help="comma-separated savestate slots rotated per episode (default: 2)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=os.path.join(MODEL_DIR, "neuro_best.npz"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--teacher-run", default="auto",
                        help="autopilot hive-mind run used to seed generation 1")
    parser.add_argument("--no-teacher-seed", action="store_true")
    parser.add_argument("--correction-run", default="auto",
                        help="hive-mind run whose safety corrections seed the policy")
    parser.add_argument("--no-correction-seed", action="store_true")
    parser.add_argument("--bc-steps", type=int, default=250,
                        help="supervised hidden-layer wall warm-up steps (default: 250)")
    parser.add_argument("--ray-connection-gain", type=float, default=2.0,
                        help="ray-input weight multiplier for one protected seed (default: 2)")
    parser.add_argument("--screen-repeats", type=int, default=DEFAULT_SCREEN_REPEATS)
    parser.add_argument("--screen-fraction", type=float, default=DEFAULT_SCREEN_FRACTION)
    parser.add_argument("--no-staged", action="store_true",
                        help="give every genome every repeat (slower)")
    parser.add_argument("--end-on-wall", action="store_true",
                        help="end an episode on wall contact, including fixed time trials")
    parser.add_argument("--wall-contact-frames", type=int,
                        default=WALL_CONTACT_FRAMES_LIMIT,
                        help="consecutive contact frames required for a wall hit")
    parser.add_argument("--elite-random-split", action="store_true",
                        # Percent signs must be doubled: argparse runs every
                        # help string through %-formatting, so a bare "50% u"
                        # reads as a %u format spec and --help dies with
                        # "TypeError: %u format: a real number is required".
                        help="next generation is 50%% elite-bred lineage plus 50%% random genomes")
    parser.add_argument("--speed-weight", type=float, default=None,
                        help="checkpoint-equivalent credit for average pace in "
                             "progress_score (default %.1f); 0 disables"
                             % genetic.SPEED_PROGRESS_BONUS)
    parser.add_argument("--curriculum-start", type=int,
                        default=DEFAULT_CURRICULUM_START)
    parser.add_argument("--curriculum-step", type=int,
                        default=DEFAULT_CURRICULUM_STEP)
    parser.add_argument("--no-curriculum", action="store_true")
    parser.add_argument("--checkpoint", default=None,
                        help="GA state file; default is OUT.state.pkl")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="resume an interrupted generation checkpoint")
    parser.add_argument("--fork", default=None,
                        help="seed a clean rules-versioned run from another GA checkpoint")
    args = parser.parse_args()

    if args.evaluation_seconds < 0:
        raise SystemExit("--evaluation-seconds cannot be negative")
    if args.speed_weight is not None:
        if args.speed_weight < 0:
            raise SystemExit("--speed-weight cannot be negative")
        genetic.SPEED_PROGRESS_BONUS = args.speed_weight
    if args.bc_steps < 0:
        raise SystemExit("--bc-steps cannot be negative")
    if args.ray_connection_gain <= 0:
        raise SystemExit("--ray-connection-gain must be positive")
    if args.wall_contact_frames < 1:
        raise SystemExit("--wall-contact-frames must be at least 1")
    evaluation_frames = (
        max(int(round(args.evaluation_seconds * EMULATED_FPS)), 1)
        if args.evaluation_seconds > 0 else None
    )

    if len(args.start_slots) > args.repeats:
        raise SystemExit(
            "--repeats must be at least the number of --start-slots so every "
            "genome is evaluated from every turn"
        )
    effective_screen_repeats = max(
        args.screen_repeats, min(len(args.start_slots), args.repeats)
    )

    if args.repeats < 2:
        print("! --repeats 1 selects for luck: identical configs here have "
              "scored 81 then 66. Continuing, but do not trust the result.\n")

    os.makedirs(MODEL_DIR, exist_ok=True)

    inputs = NUM_FEATURES * args.frame_stack
    size = genome_size(inputs, args.hidden)
    rng = np.random.default_rng(args.seed)
    picker = random.Random(args.seed)
    diversity_probes, _ = synthetic_wall_samples(
        args.frame_stack, count=DIVERSITY_PROBE_COUNT, seed=991
    )
    checkpoint_path = args.checkpoint or f"{args.out}.state.pkl"
    structural = {
        "inputs": inputs,
        "hidden": args.hidden,
        "population": args.population,
        "frame_stack": args.frame_stack,
    }
    # A fork may import weights from an older ruleset. Resume, however, must
    # match the time-trial duration so unlike fitness evidence is never mixed.
    expected = dict(
        structural,
        start_slots=tuple(args.start_slots),
        evaluation_frames=evaluation_frames,
        end_on_wall=bool(args.end_on_wall),
        wall_contact_frames=args.wall_contact_frames,
        elite_random_split=bool(args.elite_random_split),
        rules_version=EVOLUTION_RULES_VERSION,
    )

    population = None
    population_lineage = None
    history = WeightHistory()
    best = None
    best_score = None
    finalists = None
    hall_of_fame = []
    generation_start = 0
    curriculum_target = 0 if args.no_curriculum else max(args.curriculum_start, 1)
    run_id = args.run_id or new_run_id("neuro")
    seed_note = "random population"
    guidance = None
    guidance_note = "disabled"
    correction_source = None
    forked = False
    progress_best = -1.0
    last_progress_generation = 0

    resume_path = checkpoint_path if args.resume == "auto" else args.resume
    if resume_path and os.path.exists(resume_path):
        state = load_checkpoint(resume_path, expected)
        can_extend = args.generations > state.get("next_generation", 0)
        if not state.get("complete") or can_extend:
            population = [genome.copy() for genome in state["population"]]
            population_lineage = state.get("population_lineage")
            generation_start = int(state["next_generation"])
            best = state.get("best")
            best_score = state.get("best_score")
            finalists = state.get("finalists")
            hall_of_fame = state.get("hall_of_fame", [])
            run_id = state.get("run_id", run_id)
            curriculum_target = int(state.get("curriculum_target", curriculum_target))
            history._episodes = state.get("history", {})
            rng.bit_generator.state = state["rng_state"]
            picker.setstate(state["picker_state"])
            guidance = state.get("guidance")
            guidance_note = state.get("guidance_note", "checkpoint guidance")
            correction_source = state.get("correction_source")
            progress_best = float(state.get("progress_best", -1.0))
            last_progress_generation = int(
                state.get("last_progress_generation", generation_start)
            )
            seed_note = f"resumed {resume_path} after generation {generation_start}"
        else:
            population = [genome.copy() for genome in state["population"]]
            population_lineage = None
            curriculum_target = int(
                state.get("curriculum_target", curriculum_target) or curriculum_target
            )
            correction_source = state.get("run_id")
            guidance = state.get("guidance")
            guidance_note = state.get("guidance_note", "checkpoint guidance")
            forked = True
            seed_note = f"clean run forked from completed {resume_path}"
    elif args.resume and args.resume != "auto":
        raise SystemExit(f"Resume checkpoint not found: {resume_path}")

    if population is None and args.fork:
        if not os.path.exists(args.fork):
            raise SystemExit(f"Fork checkpoint not found: {args.fork}")
        state = load_checkpoint(args.fork, structural)
        population = [genome.copy() for genome in state["population"]]
        population_lineage = None
        if len(population) != args.population or any(g.size != size for g in population):
            raise SystemExit(f"Fork checkpoint {args.fork} has incompatible genomes")
        curriculum_target = int(
            state.get("curriculum_target", curriculum_target) or curriculum_target
        )
        correction_source = state.get("run_id")
        forked = True
        seed_note = f"clean rules-v{EVOLUTION_RULES_VERSION} fork of {args.fork}"

    if population is None:
        teacher_run = None if args.no_teacher_seed else args.teacher_run
        seed_model = args.out if os.path.exists(args.out) else None
        population, seed_note = initial_population(
            size,
            args.population,
            inputs,
            args.hidden,
            args.frame_stack,
            rng,
            teacher_run=teacher_run,
            seed_model=seed_model,
            allow_teacher=not args.no_teacher_seed,
        )

    if not args.no_correction_seed and guidance is None:
        requested_run = args.correction_run
        if requested_run == "auto" and correction_source:
            requested_run = correction_source
        guidance, guidance_note = fit_guidance_seed(
            requested_run, args.frame_stack, inputs, args.hidden
        )
        correction_source = requested_run

    supervised_seed = None
    ray_seed = None
    supervised_note = "not requested"
    if forked and population and args.bc_steps > 0:
        supervised_seed, supervised_note = supervised_wall_warmup(
            population[0], args.frame_stack, inputs, args.hidden,
            steps=args.bc_steps, seed=args.seed + 7331,
        )
        ray_seed = emphasise_ray_connections(
            supervised_seed, args.frame_stack, inputs, args.hidden,
            gain=args.ray_connection_gain,
        )
        supervised_note += f"; additional {args.ray_connection_gain:g}x ray-weight candidate"

    if forked:
        population = diversify_forked_population(
            population, guidance, rng, size, inputs, args.hidden,
            supervised_seed=supervised_seed,
            ray_seed=ray_seed,
        )
        # Old fitness/history belongs to different termination and ranking
        # rules.  Only the weights transfer; all evidence starts clean.
        best = None
        best_score = None
        finalists = None
        hall_of_fame = []
        history = WeightHistory()
        generation_start = 0
        progress_best = -1.0
        last_progress_generation = 0
    elif generation_start == 0 and guidance is not None and population:
        population[0] = guidance.copy()
        for index in range(1, min(1 + GUIDANCE_VARIANTS, len(population))):
            population[index] = breed(
                guidance,
                guidance,
                rng,
                size,
                inputs,
                args.hidden,
                LARGE_MUTATION_SIGMA,
            )

    if len(population_lineage or []) != len(population):
        population_lineage = default_lineage(
            population, "forked" if forked else "initial"
        )

    if evaluation_frames:
        # The fixed clock is the curriculum: everybody gets the same amount
        # of game time and is ranked by signed displacement at the end.
        curriculum_target = 0

    shape = "linear" if args.hidden <= 0 else f"1 hidden layer of {args.hidden}"
    print(f"Evolving a {shape} policy: {inputs} inputs -> {NUM_ACTIONS} actions, "
          f"{size} parameters")
    print(f"population {args.population}, {args.generations} generations, "
          f"{args.repeats} episodes per genome")
    if evaluation_frames:
        print(f"time trial: {evaluation_frames} emulated frames "
              f"({args.evaluation_seconds:g} seconds); signed net distance")
    if args.end_on_wall:
        print(f"wall ending: after {args.wall_contact_frames} contact frame(s)")
    if args.elite_random_split:
        print("population renewal: 2 exact elites + bred top-half children + "
              "50% fresh random routes")
    else:
        print("population renewal: 2 fitness elites + 2 behavior elites + "
              "10% elite variants + 8% targeted children + 20% random "
              "(50% random only after a 5-generation stall)")
    print(f"turn restart slots: {','.join(str(slot) for slot in args.start_slots)}; "
          "fitness uses progress gained from each start")
    print(f"speed weight: {genetic.SPEED_PROGRESS_BONUS:.1f} checkpoints per unit "
          "of mean pace, inside progress_score")
    if args.no_staged:
        print(f"{args.population * args.repeats} episodes per generation across "
              f"{args.instances} emulators")
    else:
        promoted = max(ELITES, int(np.ceil(args.population * args.screen_fraction)))
        budget = (
            args.population * min(effective_screen_repeats, args.repeats)
            + promoted * max(args.repeats - effective_screen_repeats, 0)
        )
        print(f"staged screening: about {budget} episodes/generation "
              f"({args.population * args.repeats - budget} saved)")
    print(f"seed: {seed_note}")
    print(f"wall/correction guidance: {guidance_note}")
    print(f"supervised hidden warm-up: {supervised_note}")
    print(f"automatic checkpoint: {checkpoint_path}")
    print(f"run '{run_id}'\n")

    final_ranked = None
    started = time.time()

    for generation in range(generation_start, args.generations):
        if args.no_staged:
            scores = score_population(
                population, args.instances, run_id, args.repeats, args.frame_skip,
                args.frame_stack, inputs, args.hidden, history, curriculum_target,
                args.start_slots, instance_offset=args.offset, generation=generation,
                evaluation_frames=evaluation_frames,
                end_on_wall=args.end_on_wall,
                wall_contact_frames=args.wall_contact_frames,
            )
            promoted_count = len(population)
        else:
            scores, promoted_count = score_population_staged(
                population,
                args.instances,
                run_id,
                args.repeats,
                args.frame_skip,
                args.frame_stack,
                inputs,
                args.hidden,
                history,
                curriculum_target,
                effective_screen_repeats,
                args.screen_fraction,
                args.start_slots,
                instance_offset=args.offset,
                generation=generation,
                evaluation_frames=evaluation_frames,
                end_on_wall=args.end_on_wall,
                wall_contact_frames=args.wall_contact_frames,
            )

        ranked = sorted(zip(population, scores), key=lambda pair: rank_key(pair[1]), reverse=True)
        final_ranked = ranked
        leader, leader_score = ranked[0]
        append_generation_summary(
            run_id,
            generation_summary(
                generation + 1, population, scores, population_lineage
            ),
        )
        finalists = [genome.copy() for genome, _ in ranked[:POLICY_FINALISTS]]
        hall_of_fame = update_hall_of_fame(hall_of_fame, ranked)

        if best_score is None or rank_key(leader_score) > rank_key(best_score):
            best, best_score = leader.copy(), dict(leader_score)
            np.savez(args.out, weights=best, hidden=args.hidden, inputs=inputs)

        leader_distance = leader_score.get("distance", leader_score["furthest"])
        if leader_distance > progress_best + 0.05:
            progress_best = float(leader_distance)
            last_progress_generation = generation
        stall_age = generation - last_progress_generation
        stalled = stall_age >= STALL_GENERATIONS

        print(
            f"gen {generation + 1:3d}/{args.generations}  "
            f"best {leader_score['fitness']:6.1f} "
            f"(distance {leader_distance:+5.1f} +-{leader_score['spread']:4.1f}, "
            f"frontier {leader_score['furthest']:5.1f} "
            f"over {leader_score['episodes']:2d} eps, "
            f"wall {leader_score['wall_rate'] * 100:4.1f}%, "
            f"centre {leader_score['centre'] * 100:4.1f}%, "
            f"speed {leader_score['speed'] * 100:4.1f}%, "
            f"risk {leader_score.get('wall_risk', 0.0) * 100:4.1f}%, "
            f"assist {leader_score.get('safety_override_rate', 0.0) * 100:4.1f}%, "
            f"agree {leader_score.get('policy_agreement', 0.0) * 100:4.1f}%, "
            f"turn {leader_score.get('turn_quality', 0.0) * 100:4.1f}%, "
            f"lean {leader_score.get('actual_lean_rate', 0.0) * 100:4.1f}%, "
            f"laps {leader_score['laps']})   "
            f"mean {statistics.mean(s['fitness'] for s in scores):6.1f}   "
            f"stage {promoted_count}/{len(population)}  "
            f"course {'full' if curriculum_target == 0 else 'cp ' + str(curriculum_target)}  "
            f"pop {'EXTINCTION' if stalled else 'normal'}  "
            f"[{(time.time() - started) / 60:.1f} min]",
            flush=True,
        )

        if generation < args.generations - 1:
            next_lineage = []
            population = build_next_population(
                ranked,
                population,
                scores,
                picker,
                rng,
                size,
                inputs,
                args.hidden,
                guidance,
                stalled,
                diversity_probes,
                args.elite_random_split,
                next_lineage,
            )
            population_lineage = next_lineage

        # Advance only after the champion reaches the current lesson.  Clear
        # history when the horizon changes: a cp-20 score must not dilute the
        # same elite's evidence when it is next asked to reach cp 25.
        if curriculum_target and leader_score.get("curriculum_rate", 0.0) >= 0.5:
            next_target = curriculum_target + max(args.curriculum_step, 1)
            curriculum_target = (
                0 if next_target >= CHECKPOINTS_PER_LAP else next_target
            )
            history = WeightHistory()

        save_checkpoint(
            checkpoint_path,
            complete=False,
            config=expected,
            next_generation=generation + 1,
            population=population,
            population_lineage=population_lineage,
            best=best,
            best_score=best_score,
            finalists=finalists,
            hall_of_fame=hall_of_fame,
            history=history._episodes,
            rng_state=rng.bit_generator.state,
            picker_state=picker.getstate(),
            run_id=run_id,
            curriculum_target=curriculum_target,
            guidance=guidance,
            guidance_note=guidance_note,
            correction_source=correction_source,
            progress_best=progress_best,
            last_progress_generation=last_progress_generation,
        )

    # Confirmation run: best-fitness-ever is the winner's curse, and it was
    # measured on this project -- the controller search reported 72.2 and the
    # top three re-ran at 67.7-68.3.
    if hall_of_fame:
        finalists = [genome for genome, _ in hall_of_fame]
    elif final_ranked is not None:
        finalists = [genome for genome, _ in final_ranked[:POLICY_FINALISTS]]
    if not finalists:
        raise SystemExit("Checkpoint has no finalists to confirm.")
    print(f"\nConfirming the top {len(finalists)} on {args.final_repeats} "
          f"fresh episodes each...")

    final_scores = score_population(
        finalists, args.instances, run_id, args.final_repeats, args.frame_skip,
        args.frame_stack, inputs, args.hidden, target_progress=0,
        start_slots=args.start_slots, instance_offset=args.offset,
        generation=args.generations,
        evaluation_frames=evaluation_frames,
        end_on_wall=args.end_on_wall,
        wall_contact_frames=args.wall_contact_frames,
    )
    confirmed = sorted(zip(finalists, final_scores), key=lambda pair: rank_key(pair[1]), reverse=True)
    for rank, (_, score) in enumerate(confirmed, 1):
        print(f"  {rank}. distance {score.get('distance', score['furthest']):+5.1f} "
              f"+-{score['spread']:4.1f}  frontier {score['furthest']:5.1f}  "
              f"wall {score['wall_rate'] * 100:4.1f}%  "
              f"centre {score['centre'] * 100:4.1f}%  "
              f"speed {score['speed'] * 100:4.1f}%  "
              f"risk {score.get('wall_risk', 0.0) * 100:4.1f}%  "
              f"assist {score.get('safety_override_rate', 0.0) * 100:4.1f}%  "
              f"agree {score.get('policy_agreement', 0.0) * 100:4.1f}%  "
              f"turn {score.get('turn_quality', 0.0) * 100:4.1f}%  "
              f"lean {score.get('actual_lean_rate', 0.0) * 100:4.1f}%  "
              f"laps {score['laps']}/{score['episodes']}")

    best, best_score = confirmed[0][0], confirmed[0][1]
    np.savez(args.out, weights=best, hidden=args.hidden, inputs=inputs)
    save_checkpoint(
        checkpoint_path,
        complete=True,
        config=expected,
        next_generation=args.generations,
        population=population,
        population_lineage=population_lineage,
        best=best,
        best_score=best_score,
        finalists=finalists,
        hall_of_fame=hall_of_fame,
        history=history._episodes,
        rng_state=rng.bit_generator.state,
        picker_state=picker.getstate(),
        run_id=run_id,
        curriculum_target=0,
        guidance=guidance,
        guidance_note=guidance_note,
        correction_source=correction_source,
        progress_best=progress_best,
        last_progress_generation=last_progress_generation,
    )

    print(f"\nbest: signed distance "
          f"{best_score.get('distance', best_score['furthest']):+.1f}, "
          f"frontier {best_score['furthest']:.1f} "
          f"+-{best_score['spread']:.1f} over {best_score['episodes']} "
          f"confirmation episodes, wall {best_score['wall_rate'] * 100:.1f}%, "
          f"centre {best_score['centre'] * 100:.1f}%, "
          f"speed {best_score['speed'] * 100:.1f}%, "
          f"risk {best_score.get('wall_risk', 0.0) * 100:.1f}%, "
          f"assist {best_score.get('safety_override_rate', 0.0) * 100:.1f}%, "
          f"agree {best_score.get('policy_agreement', 0.0) * 100:.1f}%, "
          f"turn {best_score.get('turn_quality', 0.0) * 100:.1f}%, "
          f"lean {best_score.get('actual_lean_rate', 0.0) * 100:.1f}%")
    print(f"Written to {args.out}")
    print("Drive it:\n"
          f"    .venv\\Scripts\\python.exe tools\\evaluate.py --neuro {args.out}")


def _lock_range_from_argv():
    """Pull just --instances/--offset out of sys.argv for the lock.

    A second, throwaway parser rather than splitting main() into a
    lock-acquiring wrapper plus a body: main() is a long function with a
    lot of state (checkpoint/resume, curriculum, teacher-seeding), and
    re-indenting all of it under a `with` block risked breaking something
    for a purely mechanical change. parse_known_args() ignores every
    argument it doesn't recognise, so this is safe to run before main()
    parses the real, full argument list itself.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--instances", type=int, default=8)
    parser.add_argument("--offset", type=int, default=0)
    known, _ = parser.parse_known_args()
    return known.offset, known.instances


if __name__ == "__main__":
    from training.session_lock import EmulatorSessionLock

    _offset, _instances = _lock_range_from_argv()
    with EmulatorSessionLock(
        "direct genetic policy", offset=_offset, count=max(_instances, 1)
    ):
        main()
