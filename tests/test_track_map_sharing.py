"""The shared track map must survive concurrent first use.

The genetic search drives eight environments from threads, all constructing a
BridgeClient at once. An earlier version set the "loaded" flag before the load
finished, so the first thread began loading while the others read the flag as
done and took the map while it was still None. Those clients had no map, and
progress silently never advanced -- measured at 83 of 96 episodes reporting
zero progress while the cars were driving at checkpoints 14 to 27.
"""

import os
import sys
import threading
import time

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import bridge_client


def reset_cache():
    bridge_client._TRACK_MAP = None
    bridge_client._TRACK_MAP_LOADED = False


def test_every_thread_gets_the_same_loaded_map():
    reset_cache()
    results = []
    barrier = threading.Barrier(8)

    def grab():
        barrier.wait()
        results.append(bridge_client._shared_track_map())

    threads = [threading.Thread(target=grab) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 8
    assert all(r is not None for r in results), (
        f"{sum(r is None for r in results)} of 8 threads got no map; "
        "progress would silently never advance for those environments"
    )
    assert len({id(r) for r in results}) == 1, "threads got different maps"


def test_a_slow_load_does_not_hand_out_an_unloaded_map():
    """Reproduces the race directly by making the load take real time."""
    reset_cache()
    real = bridge_client.TrackMapLoadHook if hasattr(bridge_client, "TrackMapLoadHook") else None
    import track_map

    original = track_map.TrackMap

    class Slow(original):
        def __init__(self, *args, **kwargs):
            time.sleep(0.05)
            super().__init__(*args, **kwargs)

    track_map.TrackMap = Slow
    try:
        results = []
        threads = [
            threading.Thread(target=lambda: results.append(bridge_client._shared_track_map()))
            for _ in range(6)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert all(r is not None for r in results), (
            "a thread received the map before it finished loading"
        )
    finally:
        track_map.TrackMap = original
        reset_cache()


def test_a_missing_map_fails_loudly_rather_than_scoring_everything_zero():
    reset_cache()
    import track_map

    original = track_map.TrackMap

    def boom(*args, **kwargs):
        raise FileNotFoundError("no map")

    track_map.TrackMap = boom
    try:
        bridge_client._shared_track_map()
    except RuntimeError as exc:
        assert "track map" in str(exc)
    else:
        raise AssertionError("a missing map must raise, not return None")
    finally:
        track_map.TrackMap = original
        reset_cache()


if __name__ == "__main__":
    tests = [v for n, v in sorted(globals().items()) if n.startswith("test_")]
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
