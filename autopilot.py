"""Drives the car in F-Zero (SNES) and continuously prints its distance from
the edge of the road.

Talks to BizHawk through lua/bridge.lua (see bridge_client.py):
  - receives each frame through shared memory
  - writes an action code, which bridge.lua turns into joypad.set() calls

No RAM/memory addresses are used for the edge-distance measurement -- it is
computed purely from the video frame. See vision/road_edges.py.

The bridge runs in lockstep, advancing one emulated frame per action sent,
so this loop drives the emulator forward rather than sampling it on a timer.
That means the car moves at exactly the rate this loop runs.

Usage:
    python autopilot.py                 # drive + print distances
    python autopilot.py --dry-run       # print/decide only, don't send input
    python autopilot.py --calibrate     # interactive threshold tuning, no driving
"""

import argparse
import time

import cv2

from bridge_client import (
    ACTION_NONE,
    BridgeClient,
    BridgeTimeout,
    encode_action,
)
import control
from control import SCAN_FRACTIONS, Controller
from vision.road_edges import (
    DEFAULT_CAR_HALF_WIDTH,
    DEFAULT_SAT_MAX,
    DEFAULT_V_MAX,
    DEFAULT_V_MIN,
    draw_debug_overlay,
    measure_edges,
    measure_rows,
)


# Consecutive frames with no road anywhere before calling it a crash-out.
# A couple of blank frames happen on transitions; a results screen persists.
CRASH_FRAMES = 30

# Road pixels across all scan rows below which we are not looking at a race.
# A real race view has hundreds; a results screen measured 46 strays.
MIN_ROAD_PIXELS = 120


def fmt_px(value):
    return "clear" if value is None else f"{value}px"


def make_controller(args):
    return Controller(
        deadzone=args.deadzone,
        full_scale=args.full_scale,
        lean_threshold=args.lean_threshold,
        curve_slowdown=args.curve_slowdown,
        min_corner_speed=args.min_corner_speed,
        row_weights=args.row_weights,
        target_clearance=args.target_clearance,
        allow_lean=not args.no_lean,
    )


def run_drive(args):
    controller = make_controller(args)
    bridge = BridgeClient()
    last_print = 0.0
    action = encode_action(*controller.update([], 128.0, 0)[0])
    frames = 0

    print("Driving. Ctrl+C to stop (releases all inputs).")
    if args.dry_run:
        print("--dry-run: decisions will be printed but not sent to BizHawk.")

    if args.reset:
        # Start from the savestate in slot 1 rather than wherever the
        # emulator happens to be, so a run is repeatable.
        print("Loading savestate (slot 1)...")
        bridge.reset()
        print(f"  at lap {bridge.lap}, progress {bridge.progress}/90")

    started = time.time()
    lap_started = started
    last_lap = None
    laps_done = 0
    blind_frames = 0
    crashes = 0

    log = None
    if args.log:
        log = open(args.log, "w", newline="")
        log.write(
            "frame,elapsed,lap,progress,speed,target,severity,signal,"
            "steer,throttle,shoulder,near_left,near_right,mid_left,mid_right,"
            "far_left,far_right\n"
        )

    period = 1.0 / args.hz if args.hz > 0 else 0.0

    try:
        while True:
            loop_start = time.time()

            # In dry-run we still have to advance the emulator to get a new
            # frame, so send a neutral action rather than the steering one.
            sent = ACTION_NONE if args.dry_run else action
            speed, frame = bridge.advance(sent)
            frames += 1

            if period:
                time.sleep(max(0.0, period - (time.time() - loop_start)))

            if frame is None:
                continue

            h, w = frame.shape[:2]
            readings = measure_rows(
                frame,
                [int(h * f) for f in SCAN_FRACTIONS],
                car_half_width=args.car_half_width,
                sat_max=args.sat_max,
                v_min=args.v_min,
                v_max=args.v_max,
            )

            # A results / "CRASHED OUT" screen has no road anywhere, not
            # even off to one side. Without this the controller happily
            # steers against a black screen for thousands of frames -- most
            # of two earlier runs went that way, looking like a car wedged
            # on a wall rather than one that had already been destroyed.
            # Counted, not "is there any road at all": a results screen
            # still leaves a few dozen stray pixels passing the road
            # threshold, which was enough to look like racing and let the
            # run idle against a black screen undetected.
            racing = sum(r["road_pixels"] for r in readings) >= MIN_ROAD_PIXELS

            if racing:
                blind_frames = 0
            else:
                blind_frames += 1
                if blind_frames >= CRASH_FRAMES:
                    crashes += 1
                    print(
                        f"*** CRASHED OUT at lap {bridge.lap} cp {bridge.progress} "
                        f"after {time.time() - lap_started:.1f}s (crash #{crashes}) ***"
                    )
                    if not args.retry_on_crash:
                        break

                    bridge.reset()
                    controller = make_controller(args)
                    blind_frames = 0
                    last_lap = bridge.lap
                    lap_started = time.time()
                    action = ACTION_NONE
                    continue

            if args.dump_frames and frames % args.dump_every == 0:
                import os as _os
                _os.makedirs(args.dump_frames, exist_ok=True)
                cv2.imwrite(
                    _os.path.join(
                        args.dump_frames,
                        f"cp{bridge.progress:02d}_f{frames:06d}.png",
                    ),
                    frame,
                )

            dims, reason = controller.update(readings, w / 2.0, speed)
            action = encode_action(*dims)

            if log is not None:
                def px(value):
                    return "" if value is None else value

                log.write(
                    f"{frames},{time.time() - started:.3f},{bridge.lap},"
                    f"{bridge.progress},{speed},{controller.last_target:.0f},"
                    f"{controller.last_severity:.4f},{controller._last_signal:.4f},"
                    f"{dims[0]},{dims[1]},{dims[2]},"
                    f"{px(readings[0]['left_px'])},{px(readings[0]['right_px'])},"
                    f"{px(readings[1]['left_px'])},{px(readings[1]['right_px'])},"
                    f"{px(readings[2]['left_px'])},{px(readings[2]['right_px'])}\n"
                )

            if args.max_seconds and time.time() - started > args.max_seconds:
                print(f"\nReached --max-seconds ({args.max_seconds:.0f}s).")
                break

            if last_lap is None:
                last_lap = bridge.lap
            elif bridge.lap != last_lap:
                elapsed = time.time() - lap_started
                print(
                    f"*** LAP COMPLETE: {last_lap} -> {bridge.lap} in "
                    f"{elapsed:.1f}s ***"
                )
                last_lap = bridge.lap
                lap_started = time.time()
                laps_done += 1
                if args.laps and laps_done >= args.laps:
                    print(f"Completed {laps_done} lap(s).")
                    break

            if time.time() - last_print >= args.print_interval:
                rate = frames / max(time.time() - started, 1e-6)
                near = readings[0]
                print(
                    f"lap {bridge.lap} @{bridge.progress:2d}/90  "
                    f"L={fmt_px(near['left_px']):>7} R={fmt_px(near['right_px']):>7}  "
                    f"speed {speed:5d}  [{rate:3.0f} fps]  -> {reason}"
                )
                last_print = time.time()

    except BridgeTimeout as exc:
        print(f"\n[error] {exc}")
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        if log is not None:
            log.close()
            print(f"Telemetry written to {args.log}")
        print(f"crashes: {crashes}, laps completed: {laps_done}")
        print("Stopped, released all inputs.")


def run_calibrate(args):
    bridge = BridgeClient()

    window = "F-Zero road edge calibration"
    cv2.namedWindow(window)
    cv2.createTrackbar("sat_max", window, args.sat_max, 255, lambda v: None)
    cv2.createTrackbar("v_min", window, args.v_min, 255, lambda v: None)
    cv2.createTrackbar("v_max", window, args.v_max, 255, lambda v: None)
    cv2.createTrackbar("car_half_w", window, args.car_half_width, 80, lambda v: None)
    cv2.createTrackbar("scan_y_pct", window, int(args.scan_frac * 100), 100, lambda v: None)
    cv2.createTrackbar("scan_y2_pct", window, int(args.scan_frac2 * 100), 100, lambda v: None)

    print("Calibrate mode: adjust trackbars against the live frame.")
    print("This does NOT drive the car, but it does advance the emulator so")
    print("the picture stays live. Hold a direction in BizHawk to see how")
    print("the mask/edges look on curves and near walls.")
    print("Press 'q' or Esc in the image window to quit.")

    try:
        while True:
            _, frame = bridge.advance(ACTION_NONE)
            if frame is None:
                continue

            h, w = frame.shape[:2]
            sat_max = cv2.getTrackbarPos("sat_max", window)
            v_min = cv2.getTrackbarPos("v_min", window)
            v_max = cv2.getTrackbarPos("v_max", window)
            car_half_width = cv2.getTrackbarPos("car_half_w", window)
            scan_frac = cv2.getTrackbarPos("scan_y_pct", window) / 100.0
            scan_frac2 = cv2.getTrackbarPos("scan_y2_pct", window) / 100.0

            near = measure_edges(
                frame,
                scan_y=min(int(h * scan_frac), h - 1),
                car_half_width=car_half_width,
                sat_max=sat_max,
                v_min=v_min,
                v_max=v_max,
            )
            far = measure_edges(
                frame,
                scan_y=min(int(h * scan_frac2), h - 1),
                car_half_width=car_half_width,
                sat_max=sat_max,
                v_min=v_min,
                v_max=v_max,
            )

            overlay = draw_debug_overlay(frame, [near, far])
            overlay = cv2.resize(overlay, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)
            cv2.putText(
                overlay,
                f"near L={fmt_px(near['left_px'])} R={fmt_px(near['right_px'])}",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )
            cv2.putText(
                overlay,
                f"far  L={fmt_px(far['left_px'])} R={fmt_px(far['right_px'])}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

            mask_vis = (near["mask"].astype("uint8") * 255)
            mask_vis = cv2.resize(mask_vis, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)

            cv2.imshow(window, overlay)
            cv2.imshow("road mask", mask_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    except BridgeTimeout as exc:
        print(f"\n[error] {exc}")
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--calibrate", action="store_true", help="interactive threshold tuning, no driving")
    p.add_argument("--dry-run", action="store_true", help="print/decide only, don't send input to BizHawk")
    p.add_argument("--log", metavar="FILE", help="write per-frame telemetry CSV")
    p.add_argument("--dump-frames", metavar="DIR", help="save a frame every --dump-every frames, for checking the road mask around the track")
    p.add_argument("--dump-every", type=int, default=120)
    p.add_argument("--retry-on-crash", action="store_true", help="reload the savestate and keep going after a crash-out")
    p.add_argument("--max-seconds", type=float, default=0.0, help="stop after this long (0 = run until interrupted)")
    p.add_argument("--laps", type=int, default=0, help="stop after completing this many laps (0 = unlimited)")
    p.add_argument("--reset", action="store_true", help="load the slot-1 savestate before driving, for a repeatable run")

    p.add_argument("--hz", type=float, default=0.0, help="cap the loop rate; in lockstep this throttles the emulator itself (default: 0, uncapped)")
    p.add_argument("--print-interval", type=float, default=0.5, help="seconds between console prints (default: 0.5)")

    p.add_argument("--deadzone", type=float, default=control.STEER_DEADZONE, help="below this steering demand, run straight")
    p.add_argument("--full-scale", type=float, default=control.STEER_FULL_SCALE, help="steering demand that means full lock every frame")
    p.add_argument("--lean-threshold", type=float, default=control.LEAN_THRESHOLD, help="demand above which the shoulder button is added")
    p.add_argument("--curve-slowdown", type=float, default=control.CURVE_SLOWDOWN, help="how much curvature ahead lowers the target speed")
    p.add_argument("--min-corner-speed", type=int, default=control.MIN_CORNER_SPEED, help="never target a speed below this")
    p.add_argument("--no-lean", action="store_true", help="never use the shoulder buttons")
    p.add_argument("--row-weights", type=float, nargs=3, default=list(control.ROW_WEIGHTS),
                   metavar=("NEAR", "MID", "FAR"),
                   help="how much each scan row contributes to steering")
    p.add_argument("--target-clearance", type=int, default=control.TARGET_CLEARANCE,
                   help="px gap kept from a wall when only one side is visible")

    p.add_argument("--scan-frac", type=float, default=0.82, help="near scan row as a fraction of frame height (calibrate mode)")
    p.add_argument("--scan-frac2", type=float, default=0.45, help="second scan row (calibrate mode)")
    p.add_argument("--car-half-width", type=int, default=DEFAULT_CAR_HALF_WIDTH, help="px half-width of the car sprite to skip when scanning outward")

    p.add_argument("--sat-max", type=int, default=DEFAULT_SAT_MAX, help="HSV saturation threshold for 'is road'")
    p.add_argument("--v-min", type=int, default=DEFAULT_V_MIN, help="HSV min value threshold for 'is road'")
    p.add_argument("--v-max", type=int, default=DEFAULT_V_MAX, help="HSV max value threshold for 'is road'")

    return p.parse_args()


def main():
    args = parse_args()

    if args.calibrate:
        run_calibrate(args)
    else:
        run_drive(args)


if __name__ == "__main__":
    main()
