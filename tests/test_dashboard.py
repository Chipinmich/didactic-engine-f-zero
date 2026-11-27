"""Offline checks for the dashboard's GA lineage evidence."""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from tools.dashboard import (
    improvement_panel,
    lap_leaderboard,
    lineage_panel,
    timed_laps,
    verified_laps,
)


def score(distance, progress_score, wall=0.2):
    return {
        "distance": distance,
        "furthest": distance,
        "progress_score": progress_score,
        "wall_rate": wall,
        "speed": 0.5,
        "ray_response": 0.2,
        "turn_quality": 0.6,
        "centre": 0.7,
    }


def test_lineage_panel_counts_bred_improvements_not_preserved_elites():
    generations = [
        {
            "generation": 1,
            "genomes": [
                {"genome_id": "parent", "rank": 1, "source": "initial",
                 "parent_ids": [], "score": score(10.1, 8.0)},
            ],
        },
        {
            "generation": 2,
            "genomes": [
                {"genome_id": "elite", "rank": 2, "source": "elite",
                 "parent_ids": ["parent"], "score": score(10.1, 8.0)},
                {"genome_id": "child", "rank": 1, "source": "bred",
                 "parent_ids": ["parent"], "score": score(11.1, 7.0)},
                {"genome_id": "random", "rank": 3, "source": "random",
                 "parent_ids": [], "score": score(2.0, 1.0)},
            ],
        },
    ]

    panel = lineage_panel(generations)

    assert "child" in panel
    assert "1/1" in panel
    assert "bred:1" in panel
    assert "elite:1" in panel
    assert "random:1" in panel


def episode(progress, reward, wall, assist, turn, toward, finished=False):
    return {
        "net_progress": progress,
        "reward": reward,
        "wall_rate": wall,
        "safety_override_rate": assist,
        "turn_quality": turn,
        "ray_toward_wall_rate": toward,
        "finished": finished,
    }


def test_improvement_panel_calls_out_a_broad_improvement():
    old = episode(4, -20, 0.30, 0.20, 0.55, 0.45)
    new = episode(6, 20, 0.20, 0.12, 0.65, 0.30)
    panel = improvement_panel([old] * 10 + [new] * 10, window=10)

    assert "Going in the right direction" in panel
    assert "signed progress" in panel
    assert "4.00" in panel and "6.00" in panel


def test_improvement_panel_calls_out_regression():
    old = episode(6, 20, 0.20, 0.12, 0.65, 0.30)
    new = episode(3, -20, 0.32, 0.22, 0.50, 0.48)
    panel = improvement_panel([old] * 10 + [new] * 10, window=10)

    assert "Trending in the wrong direction" in panel


def test_improvement_panel_waits_for_two_complete_windows():
    panel = improvement_panel([episode(1, 0, 0, 0, 0, 0)] * 19, window=10)

    assert "Collecting evidence" in panel
    assert "Need 20 completed episodes" in panel


def test_lap_leaderboard_requires_full_distance_and_sorts_by_frames():
    false_finish = {
        "finished": True, "laps": 1, "furthest": 60,
        "progress_gain": 60, "episode_frames": 500,
    }
    slow = {
        "finished": True, "laps": 1, "furthest": 90,
        "progress_gain": 90, "episode_frames": 1800,
        "instance": 8, "episode": 4,
    }
    fast = {
        "finished": True, "laps": 1, "furthest": 90,
        "progress_gain": 90, "episode_frames": 1200,
        "instance": 7, "episode": 3,
    }

    assert verified_laps([false_finish, slow, fast]) == [slow, fast]
    panel = lap_leaderboard([false_finish, slow, fast])
    assert "20.00s" in panel and "30.00s" in panel
    assert panel.index("20.00s") < panel.index("30.00s")
    assert "500" not in panel


def test_mixed_start_leaderboard_times_only_race_start_slot_zero():
    race_start = {
        "finished": True, "laps": 1, "furthest": 90,
        "progress_gain": 90, "episode_frames": 1800,
        "start_slot": 0, "instance": 1, "episode": 3,
    }
    corner_start = {
        "finished": True, "laps": 1, "furthest": 90,
        "progress_gain": 90, "episode_frames": 700,
        "start_slot": 7, "instance": 2, "episode": 4,
    }

    assert timed_laps([corner_start, race_start]) == [race_start]
    panel = lap_leaderboard([corner_start, race_start])
    assert "30.00s" in panel
    assert "11.67s" not in panel
    assert "only slot 0 laps" in panel


def run():
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
    print(f"{len(tests)} dashboard tests passed")


if __name__ == "__main__":
    run()
