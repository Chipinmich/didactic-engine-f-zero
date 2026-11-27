"""Unit tests for the direct no-coast genetic driving policy."""

import os
import sys
import tempfile

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from training.genetic import (
    RAY_CLEARANCE_RECOVERY_BONUS,
    RAY_RESPONSE_PROGRESS_BONUS,
    SPEED_PROGRESS_BONUS,
    rank_key,
    summarise,
)
from training.neuroevolution import (
    NUM_ACTIONS,
    Stacker,
    act,
    analytic_teacher_seed,
    action_signature,
    build_next_population,
    breed,
    diverse_parent_indices,
    evaluation_scenarios,
    emphasise_ray_connections,
    fit_balanced_linear,
    genome_size,
    lift_linear_genome,
    load_checkpoint,
    mirror_action_codes,
    mirror_stacked_observations,
    policy_ray_probe_score,
    ray_input_mask,
    save_checkpoint,
    synthetic_ray_probes,
    synthetic_wall_samples,
    supervised_wall_warmup,
    update_hall_of_fame,
    parse_start_slots,
    upgrade_legacy_stacked_observations,
)
import training.neuroevolution as neuro
from vision.road_edges import (
    LEGACY_NUM_FEATURES,
    NUM_FEATURES,
    NUM_RAY_FEATURES,
    RAY_START_INDEX,
    ROAD_LEFT_INDEX,
    ROAD_RIGHT_INDEX,
    SPEED_INDEX,
)


def test_policy_has_all_steer_and_lean_combinations():
    """The GA can choose all 3 steering x 3 lean combinations, never coast."""
    inputs = NUM_FEATURES * 8
    assert NUM_ACTIONS == 9
    assert genome_size(inputs, 0) == inputs * 9 + 9

    # The final 9 values are the linear policy's action biases. Make each
    # one win in turn and check the flattened code maps to the intended pair.
    observation = np.zeros(inputs, dtype=np.float32)
    for code in range(9):
        genome = np.zeros(genome_size(inputs, 0), dtype=np.float32)
        genome[-9 + code] = 1.0
        assert act(genome, observation, inputs, 0) == code
        assert (code // 3, code % 3) in {
            (steer, lean) for steer in range(3) for lean in range(3)
        }


def test_linear_teacher_lifts_into_32_hidden_units_without_changing_lessons():
    observations, labels = synthetic_wall_samples(2, count=200)
    inputs = observations.shape[1]
    linear, _ = fit_balanced_linear(observations, labels)
    hidden = lift_linear_genome(linear, inputs, 32)
    assert hidden.shape == (genome_size(inputs, 32),)
    linear_actions = action_signature(linear, observations, inputs, 0)
    hidden_actions = action_signature(hidden, observations, inputs, 32)
    assert np.mean(linear_actions == hidden_actions) > 0.98


def test_supervised_warmup_teaches_hidden_units_wall_to_turn_connections():
    inputs = NUM_FEATURES * 8
    rng = np.random.default_rng(12)
    genome = rng.normal(0.0, 0.02, genome_size(inputs, 32)).astype(np.float32)
    observations, labels = synthetic_wall_samples(8, count=1000, seed=14)
    before = np.mean(action_signature(genome, observations, inputs, 32) == labels)
    warmed, _ = supervised_wall_warmup(
        genome, 8, inputs, 32, steps=80, seed=14
    )
    after = np.mean(action_signature(warmed, observations, inputs, 32) == labels)

    assert after > before + 0.35
    assert after > 0.70


def test_ray_gain_only_scales_hidden_connections_from_seven_rays():
    stack, hidden = 2, 4
    inputs = NUM_FEATURES * stack
    genome = np.ones(genome_size(inputs, hidden), dtype=np.float32)
    stronger = emphasise_ray_connections(genome, stack, inputs, hidden, gain=2.0)
    incoming = stronger[: inputs * hidden].reshape(inputs, hidden)
    ray_rows = {
        age * NUM_FEATURES + feature
        for age in range(stack)
        for feature in range(RAY_START_INDEX, RAY_START_INDEX + NUM_RAY_FEATURES)
    }
    for row in range(inputs):
        assert np.all(incoming[row] == (2.0 if row in ray_rows else 1.0))
    assert np.all(stronger[inputs * hidden :] == 1.0)


def test_ray_mask_selects_all_seven_rays_in_every_stacked_frame():
    mask = ray_input_mask(NUM_FEATURES * 8)
    assert int(mask.sum()) == NUM_RAY_FEATURES * 8
    for frame in range(8):
        base = frame * NUM_FEATURES
        assert not mask[base + SPEED_INDEX]
        assert np.all(
            mask[base + RAY_START_INDEX : base + RAY_START_INDEX + NUM_RAY_FEATURES]
        )


def test_deterministic_ray_probe_rewards_actual_ray_to_turn_connections():
    stack = 2
    inputs = NUM_FEATURES * stack
    genome = np.zeros(genome_size(inputs, 0), dtype=np.float32)
    weights = genome[: inputs * NUM_ACTIONS].reshape(inputs, NUM_ACTIONS)
    bias = genome[inputs * NUM_ACTIONS :]
    bias[0] = 0.2
    for frame in range(stack):
        start = frame * NUM_FEATURES + RAY_START_INDEX
        left = slice(start, start + 3)
        right = slice(start + 4, start + 7)
        weights[left, 3] = 1.0
        weights[right, 3] = -1.0
        weights[left, 6] = -1.0
        weights[right, 6] = 1.0

    observations, targets = synthetic_ray_probes(stack)
    assert observations.shape[1] == inputs
    assert set(targets) == {0, 1, 2}
    assert policy_ray_probe_score(genome, stack, inputs, 0) == 1.0


def test_hidden_crossover_keeps_each_neuron_coherent():
    inputs, hidden = 3, 12
    size = genome_size(inputs, hidden)
    parent_a = np.zeros(size, dtype=np.float32)
    parent_b = np.ones(size, dtype=np.float32)
    child = breed(
        parent_a, parent_b, np.random.default_rng(4), size, inputs, hidden, sigma=0.0
    )
    cut = inputs * hidden
    cut2 = cut + hidden
    cut3 = cut2 + hidden * NUM_ACTIONS
    w1 = child[:cut].reshape(inputs, hidden)
    b1 = child[cut:cut2]
    w2 = child[cut2:cut3].reshape(hidden, NUM_ACTIONS)
    for unit in range(hidden):
        values = np.concatenate((w1[:, unit], [b1[unit]], w2[unit]))
        assert np.all(values == values[0]), "a hidden feature was split across parents"


def test_diverse_parent_pool_keeps_different_action_patterns_within_good_group():
    inputs = 4
    population = []
    for action in range(8):
        genome = np.zeros(genome_size(inputs, 0), dtype=np.float32)
        genome[-NUM_ACTIONS + action] = 1.0
        population.append(genome)
    scores = [
        summarise([{"furthest": value, "wall_rate": 0, "centre": 0,
                    "speed": 0, "lap": 0}])
        for value in range(8)
    ]
    probes = np.zeros((16, inputs), dtype=np.float32)
    selected = diverse_parent_indices(population, scores, probes, inputs, 0, 4)
    signatures = [tuple(action_signature(population[i], probes, inputs, 0)) for i in selected]
    assert selected[0] == 7, "fitness champion must remain first"
    assert len(set(signatures)) == 4, "parent pool collapsed to one behavior"


def test_turn_slot_parser_is_explicit_and_ordered():
    assert parse_start_slots("2,3,4") == (2, 3, 4)


def test_speed_separates_genomes_that_reach_the_same_place():
    """Pace must be worth real checkpoints, not a tiebreak that never fires.

    Speed used to enter fitness twice, both times inertly: as SPEED_BONUS at
    0.0001 (decorative against a score measured in checkpoints), and as the
    5th entry of rank_key's lexicographic tuple -- behind progress_score, a
    continuous float that essentially never ties, so it never got to break a
    comparison. Nothing in the search preferred a fast line to a slow one
    over the same ground. SPEED_PROGRESS_BONUS now sits inside progress_score
    itself, so the preference survives into the ranked number.
    """
    base = {"wall_rate": 0.0, "lap": 0, "centre": 0.5}
    slow = summarise([dict(base, furthest=40, speed=0.35)])
    fast = summarise([dict(base, furthest=40, speed=0.60)])

    assert rank_key(fast) > rank_key(slow)
    assert fast["progress_score"] - slow["progress_score"] > 3.0, (
        "the pace difference must show up as checkpoints in the ranked score"
    )


def test_progress_still_dominates_speed_at_realistic_margins():
    """The opposite failure: quick genomes that go nowhere must not win.

    Giving pace real weight risks buying rank with speed instead of ground.
    Mean speed/TOP_SPEED_RAW realistically spans about 0.35-0.60, so at
    SPEED_PROGRESS_BONUS = 15 the whole plausible pace spread is worth under
    four checkpoints, and a genuinely longer run still wins. Anything that
    raises the weight past the typical distance spread inverts this.
    """
    base = {"wall_rate": 0.0, "lap": 0, "centre": 0.5}
    farther_slower = summarise([dict(base, furthest=45, speed=0.35)])
    nearer_faster = summarise([dict(base, furthest=40, speed=0.60)])

    assert rank_key(farther_slower) > rank_key(nearer_faster)


def test_one_full_checkpoint_dominates_every_secondary_trait():
    poor_quality_farther = summarise([{
        "furthest": 41.0, "distance": 41.0, "wall_rate": 0.9,
        "centre": 0.0, "speed": 0.2, "speed_loss": 5.0,
        "ray_response": -1.0, "turn_quality": 0.0, "lap": 0,
    }])
    perfect_but_nearer = summarise([{
        "furthest": 40.9, "distance": 40.9, "wall_rate": 0.0,
        "centre": 1.0, "speed": 1.0, "speed_loss": 0.0,
        "ray_response": 1.0, "turn_quality": 1.0, "lap": 0,
    }])

    assert rank_key(poor_quality_farther) > rank_key(perfect_but_nearer)


def test_quality_decides_fractional_distance_near_ties():
    unsafe = summarise([{
        "furthest": 40.8, "distance": 40.8, "wall_rate": 0.7,
        "centre": 0.3, "speed": 0.4, "speed_loss": 4.0,
        "ray_response": -0.5, "turn_quality": 0.2, "lap": 0,
    }])
    safe = summarise([{
        "furthest": 40.2, "distance": 40.2, "wall_rate": 0.1,
        "centre": 0.8, "speed": 0.6, "speed_loss": 1.0,
        "ray_response": 0.5, "turn_quality": 0.8, "lap": 0,
    }])

    assert rank_key(safe) > rank_key(unsafe)


def test_a_stationary_genome_cannot_outrank_one_that_travels():
    """Pace is a mean, so it cannot be farmed by dawdling for more steps."""
    base = {"wall_rate": 0.0, "lap": 0, "centre": 0.5}
    travelling = summarise([dict(base, furthest=30, speed=0.30)])
    parked = summarise([dict(base, furthest=0, speed=1.0)])

    assert rank_key(travelling) > rank_key(parked)


def test_speed_loss_directly_reduces_genetic_progress_score():
    base = {"furthest": 40, "distance": 40, "wall_rate": 0.0,
            "lap": 0, "centre": 0.5, "speed": 0.5}
    steady = summarise([dict(base, speed_loss=0.0)])
    slowing = summarise([dict(base, speed_loss=2.0)])

    assert slowing["progress_score"] == steady["progress_score"] - 10.0
    assert rank_key(steady) > rank_key(slowing)


def test_signed_distance_beats_a_farther_genome_that_reversed():
    base = {"wall_rate": 0.0, "lap": 0, "centre": 0.5, "speed": 0.5}
    forward = summarise([dict(base, furthest=20, distance=8)])
    reversed_run = summarise([dict(base, furthest=40, distance=-2)])

    assert forward["distance"] == 8
    assert rank_key(forward) > rank_key(reversed_run)


def test_hall_of_fame_keeps_an_earlier_champion():
    size = genome_size(4, 0)
    champion = np.ones(size, dtype=np.float32)
    weak = np.zeros(size, dtype=np.float32)
    base = {"wall_rate": 0.0, "centre": 0.0, "speed": 0.0, "lap": 0}
    strong_score = summarise([dict(base, furthest=20, distance=20)])
    weak_score = summarise([dict(base, furthest=5, distance=5)])

    hall = update_hall_of_fame([], [(champion, strong_score)])
    hall = update_hall_of_fame(hall, [(weak, weak_score)])

    assert np.array_equal(hall[0][0], champion)


def test_generation_scenarios_change_real_start_state_but_stay_reproducible():
    first = evaluation_scenarios((2, 3, 4), 3, generation=0)
    second = evaluation_scenarios((2, 3, 4), 3, generation=1)
    assert first != second
    assert {slot for slot, _, _ in first} == {2, 3, 4}
    assert {slot for slot, _, _ in second} == {2, 3, 4}
    assert all(len(steers) == 4 for _, _, steers in first + second)
    assert evaluation_scenarios((2, 3, 4), 3, generation=1) == second


def test_staged_scenario_schedule_matches_one_complete_batch():
    whole = evaluation_scenarios((2, 3, 4), 3, generation=7)
    split = (
        evaluation_scenarios((2, 3, 4), 2, generation=7)
        + evaluation_scenarios(
            (2, 3, 4), 1, repeat_offset=2, generation=7
        )
    )
    assert split == whole


def test_turn_outcome_breaks_progress_ties_before_easy_straight_centering():
    base = {"wall_rate": 0.0, "lap": 0, "furthest": 40, "speed": 0.5}
    safe_turn = summarise([dict(base, centre=0.6, turn_quality=0.9, policy_agreement=0.8)])
    straight_line_specialist = summarise(
        [dict(base, centre=0.95, turn_quality=0.2, policy_agreement=0.1)]
    )
    assert rank_key(safe_turn) > rank_key(straight_line_specialist)


def test_mirroring_swaps_wall_geometry_steering_and_lean():
    observation = np.zeros(NUM_FEATURES * 2, dtype=np.float32)
    for frame in range(2):
        base = frame * NUM_FEATURES
        for row in range(4):
            observation[base + row * 3] = 0.1
            observation[base + row * 3 + 1] = 0.8
            observation[base + row * 3 + 2] = 1.0
        observation[base + ROAD_LEFT_INDEX] = 0.7
        observation[base + ROAD_RIGHT_INDEX] = 0.2
        observation[base + SPEED_INDEX] = 0.5
        observation[
            base + RAY_START_INDEX : base + RAY_START_INDEX + NUM_RAY_FEATURES
        ] = np.arange(NUM_RAY_FEATURES)

    mirrored = mirror_stacked_observations(observation)
    assert mirrored[0] == 0.8 and mirrored[1] == 0.1
    assert mirrored[ROAD_LEFT_INDEX] == 0.2
    assert mirrored[ROAD_RIGHT_INDEX] == 0.7
    assert mirrored[SPEED_INDEX] == 0.5, "speed must not be mirrored"
    assert mirrored[RAY_START_INDEX : RAY_START_INDEX + NUM_RAY_FEATURES].tolist() == list(
        reversed(range(NUM_RAY_FEATURES))
    )
    assert mirror_action_codes(np.array([8])).tolist() == [4]


def test_legacy_corrections_gain_neutral_ray_slots_without_losing_rows():
    stack = 2
    legacy = np.arange(stack * LEGACY_NUM_FEATURES, dtype=np.float32).reshape(1, -1)
    upgraded = upgrade_legacy_stacked_observations(legacy, stack)
    assert upgraded.shape == (1, stack * NUM_FEATURES)
    for frame in range(stack):
        old = legacy[0, frame * LEGACY_NUM_FEATURES : (frame + 1) * LEGACY_NUM_FEATURES]
        new_base = frame * NUM_FEATURES
        assert np.array_equal(
            upgraded[0, new_base : new_base + LEGACY_NUM_FEATURES], old
        )
        assert np.allclose(
            upgraded[
                0,
                new_base + RAY_START_INDEX : new_base + RAY_START_INDEX + NUM_RAY_FEATURES,
            ],
            0.5,
        )


def test_synthetic_wall_lessons_cover_both_turn_directions_and_straights():
    observations, labels = synthetic_wall_samples(8, count=200)
    assert observations.shape == (200, NUM_FEATURES * 8)
    assert set(labels // 3) == {0, 1, 2}
    assert {1, 2}.issubset(set(labels % 3)), "both lean buttons need examples"


def test_next_population_keeps_elites_guidance_and_adds_new_genomes():
    size = genome_size(4, 0)
    population = [np.full(size, value, dtype=np.float32) for value in range(6)]
    scores = [summarise([{"furthest": value, "wall_rate": 0, "centre": 0,
                         "speed": 0, "lap": 0}]) for value in range(6)]
    ranked = sorted(zip(population, scores), key=lambda pair: rank_key(pair[1]), reverse=True)
    guidance = np.full(size, 99, dtype=np.float32)
    result = build_next_population(
        ranked, population, scores, __import__("random").Random(0),
        np.random.default_rng(0), size, 4, 0, guidance, stalled=True,
    )
    assert len(result) == len(population)
    assert np.array_equal(result[0], ranked[0][0])
    assert np.array_equal(result[1], ranked[1][0])
    assert any(np.array_equal(child, guidance) for child in result)
    assert any(not any(np.array_equal(child, parent) for parent in population)
               and not np.array_equal(child, guidance) for child in result[2:])

def test_stacker_is_oldest_to_newest_and_zero_padded():
    """The evolving policy sees the same time order as the prior trainer."""
    stacker = Stacker(3, features=2)
    assert stacker.reset(np.array([1, 2], dtype=np.float32)).tolist() == [0, 0, 0, 0, 1, 2]
    assert stacker.push(np.array([3, 4], dtype=np.float32)).tolist() == [0, 0, 1, 2, 3, 4]
    assert stacker.push(np.array([5, 6], dtype=np.float32)).tolist() == [1, 2, 3, 4, 5, 6]


def test_analytic_teacher_seed_steers_away_from_close_wall():
    inputs = NUM_FEATURES * 8
    genome = analytic_teacher_seed(inputs, 8)
    observation = np.zeros(inputs, dtype=np.float32)
    newest = inputs - NUM_FEATURES
    for row in range(4):
        observation[newest + row * 3] = 0.05
        observation[newest + row * 3 + 1] = 0.80
        observation[newest + row * 3 + 2] = 1.0
    code = act(genome, observation, inputs, 0)
    assert code // 3 == 2, "close left wall should seed a right steering action"


def test_ga_checkpoint_round_trip_is_complete_and_atomic():
    expected = {"inputs": 120, "hidden": 0, "population": 2, "frame_stack": 8}
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "state.pkl")
        save_checkpoint(
            path,
            complete=False,
            config=expected,
            next_generation=7,
            population=[np.zeros(3, dtype=np.float32)],
        )
        state = load_checkpoint(path, expected)
        assert state["next_generation"] == 7
        assert not state["complete"]
        assert not os.path.exists(path + ".tmp")


def test_staged_screening_spends_extra_repeats_only_on_promoted_genomes():
    population = [np.full(3, value, dtype=np.float32) for value in range(4)]
    calls = []
    original = neuro.evaluate_population

    def fake_evaluate(genomes, instances, run_id, repeats, frame_skip, stack,
                      inputs, hidden, target_progress=0, start_slots=(),
                      repeat_offset=0, instance_offset=0, generation=0,
                      evaluation_frames=None, end_on_wall=False,
                      wall_contact_frames=neuro.WALL_CONTACT_FRAMES_LIMIT):
        calls.append((len(genomes), repeats, repeat_offset, generation))
        return [
            [
                {
                    "furthest": float(genome[0]),
                    "wall_rate": 0.0,
                    "centre": 1.0,
                    "speed": 0.5,
                    "lap": 0,
                }
                for _ in range(repeats)
            ]
            for genome in genomes
        ]

    neuro.evaluate_population = fake_evaluate
    try:
        scores, promoted = neuro.score_population_staged(
            population, 8, "test", 3, 4, 8, 120, 0,
            neuro.WeightHistory(), 20, 2, 0.75, generation=9,
        )
    finally:
        neuro.evaluate_population = original

    assert calls == [(4, 2, 0, 9), (3, 1, 2, 9)]
    assert promoted == 3
    assert scores[0]["episodes"] == 2, "screened-out genome gets no costly third run"
    assert all(score["episodes"] == 3 for score in scores[1:])


def test_elite_random_split_breeds_best_traits_and_keeps_half_fresh():
    inputs, hidden, target = 4, 0, 48
    size = genome_size(inputs, hidden)
    population = [np.full(size, value, dtype=np.float32) for value in range(target)]
    scores = [
        summarise([{"furthest": value, "distance": value, "wall_rate": 0.0,
                    "centre": 0.0, "speed": 0.0, "lap": 0}])
        for value in range(target)
    ]
    ranked = sorted(
        zip(population, scores), key=lambda pair: rank_key(pair[1]), reverse=True
    )
    lineage = []
    result = build_next_population(
        ranked, population, scores, __import__("random").Random(4),
        np.random.default_rng(4), size, inputs, hidden,
        elite_random_split=True, lineage_log=lineage,
    )

    assert len(result) == target
    assert len(lineage) == target
    assert [item["source"] for item in lineage[:neuro.ELITES]] == ["elite"] * neuro.ELITES
    assert all(item["parent_ids"] for item in lineage[neuro.ELITES : target // 2])
    assert all(item["source"] == "random" and not item["parent_ids"]
               for item in lineage[target // 2 :])
    assert all(np.array_equal(result[i], ranked[i][0]) for i in range(neuro.ELITES))
    learned = result[neuro.ELITES : target // 2]
    assert learned, "the learned half needs bred children, not only elites"
    assert all(
        not any(np.array_equal(genome, original) for original in population)
        for genome in learned
    ), "top-half genomes were copied instead of recombining their traits"
    assert any(len(np.unique(np.round(genome, 3))) > 1 for genome in learned), (
        "bred children did not combine values from different top parents"
    )
    assert all(
        not any(np.array_equal(genome, original) for original in population)
        for genome in result[target // 2 :]
    )


def test_standard_population_uses_evidence_based_adaptive_mix():
    inputs, hidden, target = 4, 0, 48
    size = genome_size(inputs, hidden)
    population = []
    for value in range(target):
        genome = np.full(size, value / 100.0, dtype=np.float32)
        genome[-NUM_ACTIONS:] = 0.0
        genome[-NUM_ACTIONS + (value % NUM_ACTIONS)] = 1.0
        population.append(genome)
    scores = [
        summarise([{"furthest": value, "distance": value, "wall_rate": 0.0,
                    "centre": 0.0, "speed": 0.0, "lap": 0}])
        for value in range(target)
    ]
    ranked = sorted(
        zip(population, scores), key=lambda pair: rank_key(pair[1]), reverse=True
    )
    probes = np.zeros((16, inputs), dtype=np.float32)
    lineage = []

    result = build_next_population(
        ranked, population, scores, __import__("random").Random(8),
        np.random.default_rng(8), size, inputs, hidden,
        diversity_probes=probes, lineage_log=lineage,
    )

    sources = [item["source"] for item in lineage]
    assert len(result) == target
    assert len(lineage) == target
    assert sources.count("elite") == 2
    assert sources.count("diversity_elite") == 2
    assert sources.count("elite_variant") == 5
    assert sources.count("bred") == 25
    assert sources.count("targeted_bred") == 4
    assert sources.count("random") == 10
    assert all(item["parent_ids"] for item in lineage if item["source"] != "random")
    assert all(not item["parent_ids"] for item in lineage if item["source"] == "random")


def test_standard_population_uses_half_random_only_after_a_stall():
    inputs, hidden, target = 4, 0, 48
    size = genome_size(inputs, hidden)
    population = [
        np.full(size, value, dtype=np.float32) for value in range(target)
    ]
    scores = [
        summarise([{"furthest": value, "distance": value, "wall_rate": 0.0,
                    "centre": 0.0, "speed": 0.0, "lap": 0}])
        for value in range(target)
    ]
    ranked = sorted(
        zip(population, scores), key=lambda pair: rank_key(pair[1]), reverse=True
    )
    lineage = []

    build_next_population(
        ranked, population, scores, __import__("random").Random(9),
        np.random.default_rng(9), size, inputs, hidden,
        stalled=True, lineage_log=lineage,
    )

    assert sum(item["source"] == "random" for item in lineage) == 24
    assert sum(item["source"] == "elite" for item in lineage) == 2


def test_random_immigrants_start_at_the_target_action_mix():
    """Fresh genomes must open at TARGET_ACTION_MIX, not all-turning.

    An earlier version boosted all six turning actions equally, which drove
    straight to 0% -- overcorrecting the 98.5%-straight collapse into the
    opposite extreme. A real racing line is mostly straight with
    corrections, so the opening tendency is a deliberate mix.

    Tolerance is wide because the mix is a property of a random population,
    and because the MLP's hidden layer contributes its own variance on a
    blank observation on top of the output bias.
    """
    inputs, hidden = 176, 32
    size = genome_size(inputs, hidden)
    rng = np.random.default_rng(11)
    blank = np.zeros(inputs, dtype=np.float32)

    pop = [
        neuro.random_immigrant(rng, size, inputs=inputs, hidden=hidden)
        for _ in range(3000)
    ]
    steer = [act(g, blank, inputs, hidden) // 3 for g in pop]
    got = np.bincount(steer, minlength=3) / len(pop)

    for index, (label, target) in enumerate(
        zip(("straight", "left", "right"), neuro.TARGET_ACTION_MIX)
    ):
        assert abs(got[index] - target) < 0.06, (
            f"{label} was {got[index]:.3f}, target {target:.2f} "
            f"(full mix {got.round(3)})"
        )

    # Right must genuinely outweigh left -- the mix is asymmetric on purpose.
    assert got[2] > got[1], f"right should exceed left, got {got.round(3)}"


def test_random_immigrants_weight_ray_inputs_more_heavily():
    """The seven rays start with stronger connections than row geometry.

    They are the most directly useful geometry but only 7 of 22 features per
    frame, so a uniformly random genome barely listens to them.
    """
    inputs, hidden = 176, 32
    size = genome_size(inputs, hidden)
    rng = np.random.default_rng(3)
    stack = inputs // NUM_FEATURES

    ray_rows = [
        age * NUM_FEATURES + feature
        for age in range(stack)
        for feature in range(RAY_START_INDEX, RAY_START_INDEX + NUM_RAY_FEATURES)
    ]
    other_rows = [i for i in range(inputs) if i not in set(ray_rows)]

    genomes = [
        neuro.random_immigrant(rng, size, inputs=inputs, hidden=hidden)
        for _ in range(200)
    ]
    w1 = np.stack([g[: inputs * hidden].reshape(inputs, hidden) for g in genomes])
    ray_scale = float(np.abs(w1[:, ray_rows, :]).mean())
    other_scale = float(np.abs(w1[:, other_rows, :]).mean())

    ratio = ray_scale / other_scale
    assert abs(ratio - neuro.RAY_WEIGHT_GAIN) < 0.15, (
        f"ray inputs should start about {neuro.RAY_WEIGHT_GAIN}x the other "
        f"inputs, measured {ratio:.2f}x"
    )

    # Without the shapes the emphasis must be skipped, not guessed at.
    plain = neuro.random_immigrant(np.random.default_rng(3), size)
    w1_plain = plain[: inputs * hidden].reshape(inputs, hidden)
    assert abs(
        float(np.abs(w1_plain[ray_rows, :]).mean())
        - float(np.abs(w1_plain[other_rows, :]).mean())
    ) < 0.02, "without inputs/hidden the genome must be left unscaled"

def test_turning_bias_scales_against_the_elite():
    """The boost is measured against the elite's own straight-vs-turn gap."""
    inputs, hidden = 176, 0
    size = genome_size(inputs, hidden)

    # An elite that strongly prefers straight, like the real measured one.
    elite = np.zeros(size, dtype=np.float32)
    elite[-NUM_ACTIONS:-6] = 1.125
    expected = min(
        neuro.TURN_IMMIGRANT_MULTIPLIER * 1.125, neuro.TURN_IMMIGRANT_CAP
    )
    assert abs(neuro.turning_bias_for_immigrants(elite) - expected) < 1e-6

    neutral = np.zeros(size, dtype=np.float32)
    assert neuro.turning_bias_for_immigrants(neutral) == 0.0
    assert (
        neuro.turning_bias_for_immigrants(None) == neuro.DEFAULT_TURN_IMMIGRANT_BIAS
    )

    # Bounded, however lopsided the elite becomes.
    extreme = np.zeros(size, dtype=np.float32)
    extreme[-NUM_ACTIONS:-6] = 50.0
    assert neuro.turning_bias_for_immigrants(extreme) == neuro.TURN_IMMIGRANT_CAP


def test_no_teacher_seed_also_skips_the_analytic_fallback():
    """--no-teacher-seed must mean NO demonstration seed at all.

    It used to only skip the recorded-autopilot seed, then fall through to
    analytic_teacher_seed -- the hand-written controller, which drives
    straight until it meets a wall. That seeded up to six genomes with a
    straight-line driver, so the lineage kept re-entering the population
    even when the autopilot run was explicitly excluded.
    """
    inputs, hidden, stack = 176, 32, 8
    size = genome_size(inputs, hidden)
    blank = np.zeros(inputs, dtype=np.float32)

    seeded, seeded_note = neuro.initial_population(
        size, 48, inputs, hidden, stack, np.random.default_rng(0),
        teacher_run=None, seed_model=None, allow_teacher=True,
    )
    clean, clean_note = neuro.initial_population(
        size, 48, inputs, hidden, stack, np.random.default_rng(0),
        teacher_run=None, seed_model=None, allow_teacher=False,
    )

    # Assert on BEHAVIOUR, not wording -- the clean note legitimately names
    # the analytic fallback while saying it is disabled.
    assert "analytic controller fallback" in seeded_note, seeded_note
    assert clean_note.startswith("no demonstration seed"), clean_note


    # The property is "no demonstration genome was injected", NOT "no genome
    # ever picks straight" -- immigrants now deliberately open at
    # TARGET_ACTION_MIX, which is ~50% straight. Compare against the actual
    # analytic seed instead.
    analytic = neuro.analytic_teacher_seed(inputs, stack)
    if hidden > 0:
        analytic = neuro.lift_linear_genome(analytic, inputs, hidden)
    analytic = np.asarray(analytic, dtype=np.float32)

    def contains_analytic(population):
        return any(
            g.shape == analytic.shape
            and float(np.abs(g - analytic).max()) < 0.05
            for g in population
        )

    assert contains_analytic(seeded), (
        "the old path should demonstrably inject the analytic controller"
    )
    assert not contains_analytic(clean), (
        "a clean population must not contain the analytic straight-line seed"
    )


def test_wall_contact_costs_a_genome_real_progress():
    """Grinding a wall must not buy distance cheaply.

    In time-trial mode wall contact does not end the episode, so before this
    a genome could wedge itself against a barrier and grind out the whole
    trial at no cost. Selection now ranks on wall-adjusted distance.

    Note this lives in the GA's fitness, not env PENALTY_WALL_CONTACT: the
    genetic search never reads the env reward at all.
    """
    def episodes(distance, wall_rate):
        return [{
            "distance": distance, "furthest": distance, "wall_rate": wall_rate,
            "centre": 0.5, "speed": 0.3, "wall_risk": 0.1, "lap": 0,
            "safety_override_rate": 0.0, "policy_agreement": 1.0,
            "actual_lean_rate": 0.0, "frontier_clearance": 0.5,
            "frontier_centre": 0.5, "turn_recovery": 0.5,
            "safe_frontier_rate": 0.5, "turn_quality": 0.5, "curriculum": 0,
        } for _ in range(4)]

    clean = summarise(episodes(50.2, 0.05))
    grinder = summarise(episodes(50.8, 0.55))

    # The grinder travels fractionally farther within the same checkpoint
    # band, but spends half its trial scrubbing.
    assert grinder["distance"] > clean["distance"]
    assert clean["progress_score"] > grinder["progress_score"], (
        f"clean {clean['progress_score']:.1f} should beat grinder "
        f"{grinder['progress_score']:.1f}"
    )
    assert rank_key(clean) > rank_key(grinder), (
        "selection must prefer the cleaner line"
    )

    # A genome that never touches a wall pays no WALL penalty. The only gap
    # from raw distance is the pace credit every genome earns.
    spotless = summarise(episodes(50.0, 0.0))
    assert abs(
        spotless["progress_score"]
        - spotless["distance"]
        - SPEED_PROGRESS_BONUS * 0.3
    ) < 1e-9


def test_ray_terms_are_weighted_against_their_measured_scale():
    """The two ray weights differ hugely because the signals do.

    Measured over 1163 episodes, ray_response_score spans about 0.86 between
    its 10th and 90th percentiles while mean_ray_clearance_recovery spans
    0.013 -- sixty times smaller. Equal-looking weights therefore do not mean
    equal influence: at 10, clearance recovery was worth a tenth of a
    checkpoint and was effectively absent from selection.

    This pins the influence rather than the constants, so re-tuning stays
    honest and nobody "corrects" the large one downward for looking odd.
    """
    response_span, clearance_span = 0.86, 0.013
    response = RAY_RESPONSE_PROGRESS_BONUS * response_span
    clearance = RAY_CLEARANCE_RECOVERY_BONUS * clearance_span

    assert response > 6.0, "ray response should be a substantial term"
    assert clearance > 1.0, (
        f"clearance recovery is worth {clearance:.2f} checkpoints, which "
        "selection cannot see"
    )
    # Geometry proxies must not out-vote actually covering ground; the
    # distance spread over the same episodes was 23 checkpoints.
    assert response + clearance < 23.0, (
        "ray geometry must not outweigh distance, or a genome can win by "
        "holding a tidy line while travelling less far"
    )


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
        except AssertionError as exc:
            failures += 1
            print(f"  FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    raise SystemExit(bool(failures))
