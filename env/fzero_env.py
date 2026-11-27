import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from bridge_client import (
    ACTION_DIMS,
    CHECKPOINTS_PER_LAP,
    SHOULDER_NONE,
    STEER_LEFT,
    STEER_RIGHT,
    STEER_STRAIGHT,
    THROTTLE_ACCEL,
    THROTTLE_COAST,
    TOP_SPEED_RAW,
    BridgeClient,
    decode_action,
    encode_action,
)
from vision.road_edges import (
    NUM_FEATURES,
    SCAN_ROWS,
    features_from_readings,
    measure_rows,
    ray_distances,
    touching_wall,
)
from env.driving_assist import DrivingAssist
from track_map import LOOK_AHEAD as MAX_PLAUSIBLE_STEP

# Raw value at RAM 0x000B20, used to scale the speed feature into [0, 1].
#
# Was 5000.0 as an uncalibrated placeholder -- the comment here used to say
# so. On real recorded driving that made the feature top out at 0.17: the
# machine's true peak (TOP_SPEED_RAW, measured in bridge_client.py and
# already used there to scale the autopilot's target speed) is 2100, less
# than half of 5000. The speed input was landing in a sliver of its
# available range for no reason. Sharing the same constant both files already
# had keeps this from drifting out of sync with it again.  Seven ray features
# were later appended; the speed field remains at its explicit named index.
MAX_SPEED_RAW = float(TOP_SPEED_RAW)

# Reward is driven by progress round the track, not raw speed. Speed alone
# is farmable -- the agent can be rewarded for going fast in the wrong
# direction, or for holding speed while stuck against a wall.
#
# NOTE ON POWER: the power bar is deliberately NOT rewarded. Power is a
# consequence of hitting walls, not an achievement, and paying for a full
# bar would reward a machine that crawls round untouched. The damage is
# penalised instead, via PENALTY_WALL_CONTACT. vision/hud.py reads power
# for diagnostics only and the env never consults it.
# Raised from 30 because progress was not outweighing the shaping terms it
# competes with. Measured over 27145 potential-mode episodes, grouping episodes
# by the checkpoints they actually gained:
#
#   spread of the bucket MEANS (what progress buys)      35.8
#   spread WITHIN a bucket (obtainable without progress) 85.9
#
# So there was 2.4x more reward available from driving tidily at the same
# distance than from covering more ground, and it showed: gaining a whole
# checkpoint from 2 to 3 changed mean reward by -0.2, and a top-decile episode
# at gain 4 scored 79 against 40 for a median episode at gain 5. Luck and proxy
# quality beat distance, which is the same substitution that made the lap timer
# and WALL_RATE_PENALTY=40 fail, one level down.
#
# At 90, with PENALTY_SPEED_DECREASE_PER_UNIT cut alongside, that ratio inverts
# to 4.09 and a MEDIAN episode one checkpoint further beats a TOP-DECILE
# episode one short, at every distance measured.
#
# Safe against the two documented bounds: REWARD_STEP_COST's ceiling scales
# with this term so raising it only loosens that, and the potential bound
# (POTENTIAL_SCALE * CHECKPOINTS_PER_LAP < REWARD_LAP_COMPLETE) does not
# involve it. Progress still pays only for NEW furthest ground, so it cannot
# be farmed by rocking over a checkpoint boundary.
REWARD_PER_CHECKPOINT = 90.0

# Speed term applied to each checkpoint gained, so a fast line through the
# same checkpoints scores better than a slow one.
REWARD_SPEED_SCALE = 0.001

# Speed is also scored every step, not only on the frames a checkpoint
# ticks over, so there is a dense signal to climb rather than a sparse one
# every ~40 frames.
#
# Expressed as a SLOWNESS PENALTY: (speed - top speed) * rate, so it is
# always <= 0 and rises toward zero as the car speeds up. Going faster is
# therefore always worth more, per step and over a lap.
#
# It cannot be a positive per-step bonus. Any per-step payout is collected
# once per step, so a slow lap banks it more times than a fast one and
# scores higher -- measured 3630 against 2447, exactly backwards. Clipping
# it at zero has the same defect above the baseline, and also makes the
# term flat (so speed stops mattering) below it.
#
# The rate is deliberately small. At 0.003 this term was 47% of an
# episode's reward against 4% for progress, which buried the signal the
# agent is meant to learn from.
# Baselined on what the machine can ACTUALLY reach, not the theoretical peak.
#
# This was TOP_SPEED_RAW (2100), which the machine never gets near: measured
# peak on a clean held-throttle run is ~1575. So the term sat at roughly
# -1.39 a step permanently, with almost no gradient -- it did not distinguish
# 1000 from 1500, it just taxed existing, which is one of the things that made
# ending the episode attractive. 1600 is just above the measured ceiling, so
# the penalty still cannot become a farmable positive payout while actually
# discriminating across the speeds the machine reaches.
SPEED_REWARD_BASELINE = 1600.0
REWARD_SPEED_PER_STEP = 0.0004

# Immediate penalty for shedding speed while acceleration is pinned on.
# This is deliberately much stronger than the bounded slowness shaping above:
# a normal 4-unit dip costs 1 point, while an 80-unit coast-sized loss costs
# 20. The genetic search does not rank env reward, so the same measurement is
# also exported through info and applied in training/genetic.py.
# Cut from 0.25. At 0.25 this was the largest non-progress driver of episode
# reward -- standard deviation 60 against progress's 145 -- so a policy could
# gain more by never shedding speed than by travelling further. Lowering it
# cuts the within-bucket spread from 85.9 to 57.1.
PENALTY_SPEED_DECREASE_PER_UNIT = 0.10

# Ray-response shaping. The policy is judged only when the left/right ray
# fans disagree enough to imply a meaningful escape direction. Steering
# toward the closer wall is nearly as costly as touching it; opening the
# forward fan earns a small, event-based recovery credit. Speed-loss cost is
# reduced during these danger frames so a necessary evasive turn is not
# punished like needless weaving on a clear straight.
RAY_DANGER_CLEARANCE = 0.60
RAY_IMBALANCE_MIN = 0.08
PENALTY_STEER_TOWARD_WALL = 4.0
REWARD_RAY_CLEARANCE_GAIN = 8.0
DANGER_SPEED_LOSS_MULTIPLIER = 0.15

# Scraping a wall is what actually kills the machine -- power drains on
# contact until it is destroyed, which is how every run so far has ended.
# Charged per frame of contact, and deliberately the largest negative term.
#
# SUSTAINED wall contact ends the episode (see wall_hit in _one_frame), but a
# brief scrape gets a small recovery window. Kept per-frame rather than
# flattened to a fixed penalty so longer contact always costs more.
#
# Raising this alone does not work: a stationary machine never touches
# anything, so at 6.0 with the old progress reward, sitting still scored
# better than driving (-1.46/step against -1.63). The checkpoint reward was
# raised alongside it to keep moving worthwhile. Anything that increases
# this must re-check that -- and re-check it against the NEW termination
# rule too: test_driving_then_hitting_a_wall_beats_sitting_still in
# tests/test_reward.py is the one that would catch this regressing.
PENALTY_WALL_CONTACT = 6.0

# Being off-centre costs, in proportion to how far off-centre it is.
#
# Charged as a PENALTY, never paid as a bonus, and that is not a stylistic
# choice -- it is the trap this file already documents twice. Any positive
# per-step payout is collected once per step, so a slow lap banks it more
# times than a fast one and scores higher. A "stay centred" bonus is exactly
# that shape, and it has a second failure on top: the highest-scoring
# behaviour becomes creeping down the middle of a straight forever, which is
# both centred and going nowhere.
#
# As a penalty it is bounded above by zero, so there is nothing to farm --
# the best a machine can do is stop losing points, and it still has to
# collect the progress reward to score at all.
#
# 0 means perfectly centred, 1 means hard against a wall. Measured on real
# driving, the autopilot sits around 0.3-0.4 off-centre, so this costs it
# roughly 0.5 a frame against the ~6 a wall touch costs -- present enough to
# shape the line, small enough not to bury progress.
PENALTY_OFF_CENTRE = 1.5

# Room to leave beside a wall when only ONE wall is in frame, in pixels.
# Matches control.py's TARGET_CLEARANCE, which solves the same problem for
# the hand-written controller: with the road running off screen one way
# there is no midpoint to aim at, so "centred enough" becomes "not hugging
# the wall you can see".
REFERENCE_CLEARANCE = 55.0

# Paid once when the machine is destroyed. Large enough that no amount of
# speed or progress makes crashing worthwhile.
PENALTY_CRASH = 400.0

# Losing ground is worse than standing still, or reversing at speed becomes
# a way to farm the per-step speed reward.
PENALTY_PER_CHECKPOINT_LOST = 15.0

# Whether the race is live now comes from the game's own state, published by
# lua/bridge.lua as bridge.racing. It is NOT read off the picture any more.
#
# The picture cannot answer it. The "1. MUTE CITY I" course card is the same
# flat road-grey as the track, so counting road pixels reads it as a wide
# open road and the agent goes on steering against a menu; measured over 581
# captured frames, the "YOU LOST" screen sits at HUD-band contrast 65.9,
# squarely inside the racing distribution. Genuine racing frames were also
# falling under the old MIN_ROAD_PIXELS threshold of 120 -- one measured 92 --
# so the test cut both ways.
#
# Road pixels are still measured, but only as diagnostics.
#
# A couple of frames of confirmation, because state.txt is read while Lua may
# be rewriting it. A torn read falls back to the cached value rather than a
# wrong one, so this is belt and braces rather than load-bearing.
NOT_RACING_CONFIRM_FRAMES = 3

# KNOWN LAG: the flag drops when the menu appears, not when the machine is
# destroyed, so the ~70-frame "YOU LOST" explosion still reads as racing --
# about 17 policy steps at frame-skip 4. Searched for a byte that falls at
# the moment of death instead and there is none: across 3000 captured frames
# not one address is constant while racing and never that value during the
# explosion. The sequence passes through racing-like values.
#
# Those frames are harmless-ish (the machine is wreckage, progress has
# stopped, and _no_progress_count is already counting), but they are real
# and they do enter the recorded data. Filter on info["racing"] when
# building a training set from a run.

REWARD_PARTS = (
    "progress", "speed", "speed_loss", "ray_response", "ray_clearance",
    "wall", "centre", "crash", "lost", "step_cost", "lap", "potential",
)

# ---------------------------------------------------------------------------
# POTENTIAL-BASED SHAPING (reward_mode="potential")
#
# The legacy reward is a stack of hand-weighted proxies, and measurement says
# the agent optimised the proxies rather than the goal. Over 4530 DQN
# episodes, reward improved by +70 while distance covered did not move at all
# (progress_gain 3.21 -> 3.79, episode length 23 -> 25). 43% of that
# improvement came from the ray_response penalty alone and 23% from
# speed_loss; only 13% came from progress.
#
# Worse, it paid the agent to die. 99.8% of episodes ended on a wall hit, and
# a wall hit costs NOTHING terminal -- PENALTY_CRASH fires on `crashed`, which
# is the race actually ending, and `crash` was 0.0 on every one of those 4530
# episodes. Meanwhile continuing to drive cost -5.53 a step: a step had to
# earn 10.24 to break even and progress paid 4.71. Under any discount, ending
# the episode was the best available move.
#
# Ng, Harada and Russell (1999) give the fix, and it is the same fix for both
# problems. Adding F(s,s') = gamma*PHI(s') - PHI(s) to any reward leaves the
# optimal policy unchanged, for ANY potential PHI -- so it is a free dense
# signal that cannot be farmed. The invariance requires PHI(terminal) = 0 in
# episodic tasks, and THAT is exactly the missing terminal cost: ending an
# episode at track position p now costs -PHI(p) automatically, derived rather
# than hand-tuned.
#
# PHI is track position, scaled so one checkpoint is worth POTENTIAL_SCALE.
# Position, not the furthest-reached frontier: a potential must be a function
# of the current state. That also makes rocking back and forth over a
# checkpoint boundary worth exactly zero, because the terms telescope.
# BOUNDED BY THE LAP BONUS. Finishing zeroes the potential like any other
# termination, so a lap must pay more than the potential it gives up:
#
#   POTENTIAL_SCALE * CHECKPOINTS_PER_LAP  <  REWARD_LAP_COMPLETE * slowest bonus
#   5.0 * 90 = 450                         <  500.0 * 1.0
#
# The lap length has to be the FULL 90, not the ~82 a slot-2 start actually
# covers, or the bound holds only for the start positions in use today. At
# 12.0 the inequality fails badly (1080 > 500) and completing a lap scored
# -3727 against -33 for crashing on the spot. A test pins the inequality.
POTENTIAL_SCALE = 5.0

# Must match the LEARNER's discount factor, or the shaping stops being
# policy-invariant. sb3's DQN and PPO both default to 0.99.
POTENTIAL_GAMMA = 0.99

# Terms dropped in potential mode, and why. Each was a proxy for something the
# potential now signals directly, and each was measurably being optimised in
# preference to the objective:
#   ray_response  -2.60/step, the single largest penalty, 43% of all learning.
#                 Wall contact is already charged directly AND ends the
#                 episode, so this was the third count of the same thing.
#   centre        -1.44/step. Pure geometry proxy for "on a good line".
#   step_cost     -0.64/step. This is the term that made existing expensive,
#                 and lap time is now expressed through gamma instead: a
#                 checkpoint reached sooner is discounted less.
# speed keeps its slowness penalty but re-baselined -- see
# SPEED_REWARD_BASELINE.
POTENTIAL_MODE_DROPS = ("ray_response", "centre", "step_cost")

# Every step costs a little, which is what makes it race rather than cruise:
# a slower lap accumulates more of this penalty. This is the main lap-time
# knob. It has a ceiling though -- push it high enough that a full lap costs
# more than finishing pays, and the best move becomes crashing immediately
# to stop the meter, so keep it well under
# (LAPS_TO_FINISH * CHECKPOINTS_PER_LAP * REWARD_PER_CHECKPOINT
#  + REWARD_LAP_COMPLETE) / expected_steps_per_lap.
REWARD_STEP_COST = 0.2

# Paid once for crossing the finish line, big enough to dominate.
REWARD_LAP_COMPLETE = 500.0

# The lap bonus is scaled by how quickly the lap was done: a lap taking
# REFERENCE_LAP_FRAMES earns it in full, a faster one earns more, a slower one
# less. Without this, finishing at any speed pays the same and there is nothing
# pulling lap times down once the agent can get round at all.
#
# In EMULATED FRAMES, because _step_count is incremented in _one_frame. This
# was 4000 while a recorded human lap takes 1905 frames, so every achievable
# lap sat on LAP_SPEED_BONUS_MAX and the timing pressure did nothing at all.
# 1905 is the measured human lap, so matching a competent human pays 1.0 and
# beating it pays more, up to the cap.
REFERENCE_LAP_FRAMES = 1905
LAP_SPEED_BONUS_MIN = 0.5
LAP_SPEED_BONUS_MAX = 2.5

LAPS_TO_FINISH = 1

# Going backwards over the start line wraps the progress counter, so a
# decrease of more than this is a lap change rather than a reversal.
PROGRESS_WRAP_THRESHOLD = CHECKPOINTS_PER_LAP // 2

# Emulated frames each action is held for. The machine does not react
# within a sixtieth of a second, so acting on every frame spends experience
# without buying control: at frame_skip 1 an hour of training is only ~90
# laps, against ~356 at 4.
DEFAULT_FRAME_SKIP = 4

# Minimum frames a TURN is held for, even when the assist would hand control
# back sooner. Straight-ahead steps are unaffected.
#
# A turn needs time in the game before it does anything useful. Measured by
# holding one direction from a fixed savestate and comparing against holding
# straight, the machine is pulled off the straight line by:
#
#     frames held      5     10     15     20     30     40     59
#     units off line  35     90    152    186    215    263    412
#
# against 567 units travelled forward over the same 59 frames. So a couple of
# frames of steering is nearly nothing, and the policy was spending most of
# its decisions on holds that short: before the assist thresholds were fixed,
# 91.9% of decisions lasted a single frame, which is what made the input
# visibly flicker instead of turning the machine.
#
# 8 is about 0.13 seconds: steering is a digital SNES input, so a stronger
# turn means holding the full input longer rather than scaling its value.
# This gives a chosen turn roughly one third more authority than the former
# six-frame hold. An emergency still breaks out immediately, so the stronger
# commitment cannot keep driving toward a wall the vision system can see.
TURN_COMMITMENT_FRAMES = 8

# A warm-up gives the agent the same well-conditioned start every episode:
# centred between the walls and already at speed. Bounded so a bad start
# cannot hang an episode.
WARMUP_MAX_FRAMES = 400

# How unequal the room either side may be and still count as centred, as a
# fraction of the road width at that row.
WARMUP_CENTRE_TOLERANCE = 0.18

# Medium pace: fast enough that the handling matters, slow enough that a
# mistake is recoverable. Top speed measured about 2100.
DEFAULT_WARMUP_SPEED = 1100

# No progress for this many EMULATED FRAMES means stuck on a wall or
# wrecked. The game reports no explicit crash signal, so this stands in for
# one -- and unlike the old speed-based check it also catches a machine
# pinned against a wall at speed, which reads as moving but is going
# nowhere.
#
# Counted in frames, not policy steps: the counters live in _one_frame,
# which runs frame_skip times per step. Setting these in step units made
# every episode end after 2.5 seconds once frame-skip was introduced.
# The machine must keep MOVING. Acceleration is always held, so the only way
# to be slow is to be scrubbing against something -- and in time-trial mode
# (--evaluation-seconds) wall contact does NOT end the episode, so a genome
# that wedges itself into a wall simply grinds there for the full trial at no
# cost. Measured on a live generation: mean wall contact 54.8% of frames and
# mean speed 373 of a 2100 top speed, i.e. most of the population crawling
# along a wall rather than driving.
#
# Below this speed for this many consecutive frames the episode ends. In a
# time trial, ending early IS the penalty: distance stops accumulating.
#
# 400 sits well under real driving (the autopilot cruised 715-780) but well
# above the 240-373 a wedged machine settles at, so it separates "slowed for
# a corner" from "stopped against a wall". One second of it is deliberate --
# a brief scrub while recovering should survive.
MIN_SPEED_FLOOR = 400
MIN_SPEED_FRAMES_LIMIT = 60

# Emulated frames without reaching new ground before the episode ends.
#
# Lowered from 600 (ten seconds) purely for evaluation throughput. A lap is
# roughly 1900 frames over 90 checkpoints, about 21 frames a checkpoint, and
# the whole trial is 1800 frames -- so ten seconds of no new ground let a
# wedged genome hold an emulator for a third of its trial while returning no
# information about how it drives.
#
# 240 frames is four seconds, still more than ten checkpoints' worth of time.
# A machine that reaches no new ground in that span is stuck or going
# backwards, not taking a corner carefully. In a time trial ending early is
# itself the penalty, because distance stops accumulating.
NO_PROGRESS_FRAMES_LIMIT = 240
MAX_EPISODE_FRAMES = 20000

# Consecutive contact frames that end the episode. The 6-point per-frame
# penalty starts on the FIRST contact frame either way; this only decides
# when contact becomes fatal.
#
# 3 frames -- about 0.05 seconds -- long enough to reject a one-frame
# road-mask glitch, short enough that contact ends the run almost at once.
#
# THE PREVIOUS A/B TABLES HERE ARE VOID AND HAVE BEEN REMOVED. They compared
# limits of 3, 45 and 0 on "median distance gained" and concluded 45 nearly
# tripled reachable distance. That distance came from RAM 0x000C1, which is
# the lap TIMER, not track position -- so those tables were comparing how long
# each arm survived, and of course the arm that ended episodes latest "went
# furthest". The conclusion was an artefact of the metric, not a finding about
# driving. Nothing about 45 survives that.
#
# 3 is chosen for a structural reason rather than a measured one: it makes
# wall-grinding impossible instead of merely expensive. In time-trial mode
# contact does not otherwise end an episode, so a genome could wedge itself
# against a barrier and bank the whole trial; WALL_RATE_PENALTY existed to
# tax that. Ending on contact removes the behaviour at the source, which lets
# the fitness go back to rewarding distance rather than being dominated by a
# wall term (at 69% contact that term was worth -28 checkpoints against a
# mean distance of 26, so the search was ranking "touched walls least").
#
# THE RISK, stated plainly: short episodes can stop distinguishing genomes.
# At a limit of 1 the search went fully degenerate -- progress_gain collapsed
# into a few per-start-slot integers and 17 generations produced no slope,
# because every episode ended at whatever corner came first, so fitness
# measured track geometry rather than driving. Those numbers were also on the
# timer, but the MECHANISM is real and 3 is much nearer 1 than 45 is.
#
# What to watch, and it is checkable in one query: if progress_gain becomes
# quantised -- most episodes landing on a handful of integers, and the spread
# within a start slot collapsing -- then episodes are ending too early to rank
# genomes and this should go back up. A healthy signal is a continuous spread
# of gains within each slot.
#
# Two independent backstops remain either way: NO_PROGRESS_FRAMES_LIMIT above,
# and the game's own destruction flag. Set to 0 to disable contact-termination
# and rely only on those.
WALL_CONTACT_FRAMES_LIMIT = 3


def off_centre(readings):
    """How far across the road the machine sits, 0 centred to 1 at a wall.

    Read from the NEAR row only -- SCAN_ROWS runs far to near, so that is
    the last one, at the machine's own height. The lookahead rows are
    further down the track, where perspective squeezes the road into a
    narrow band and a perfectly placed machine still measures well off
    centre; scoring those would penalise it for the road's shape rather than
    its own position. Same reasoning as touching_wall().

    Normalised by the road width at that row, so a wide road and a narrow
    one are judged the same way.

    Both walls are often OUT OF FRAME on the near row -- the road is widest
    there, and the screen is only 256 pixels across -- so a measure that
    needs both edges reads zero on most real frames. Measured on live
    driving, a first version of this returned exactly 0.0 for entire
    episodes while the machine was demonstrably wandering into walls. So:

      both edges visible   the honest measure, position between the walls
      one edge visible     the road leaves the frame the other way, so there
                           is no midpoint to measure against. Fall back to
                           clearance: hugging the wall that IS visible is
                           off-centre, and anything past REFERENCE_CLEARANCE
                           from it counts as fine
      neither              rows are searched near to far for one that can
                           answer; if none can, 0.0

    Rows run far to near, so they are walked in reverse -- the near row is
    the machine's own position and the most relevant, and lookahead rows are
    used only when it cannot answer. Perspective squeezes the road at
    distance, so a far row exaggerates how off-centre things are; it is a
    fallback, not a preference.
    """
    for reading in reversed(readings or []):
        left, right = reading["left_px"], reading["right_px"]

        if left is not None and right is not None:
            width = left + right
            if width <= 0:
                # Wedged or off the track. The wall penalty covers that;
                # double-charging here only makes the two terms redundant.
                return 0.0
            return min(abs(right - left) / width, 1.0)

        visible = left if right is None else right
        if visible is not None:
            closeness = (REFERENCE_CLEARANCE - visible) / REFERENCE_CLEARANCE
            return min(max(closeness, 0.0), 1.0)

    return 0.0


def ray_response(ray_values, steer):
    """Return (score, opportunity, toward_wall, forward_clearance).

    Rays are ordered left-to-right. A close left fan calls for right steer,
    and vice versa. Correct/straight/wrong score +1/0/-1 respectively. Clear
    or nearly symmetric views are not scored, preventing a straightaway from
    rewarding arbitrary left/right motion.
    """
    rays = np.asarray(ray_values, dtype=np.float32).ravel()
    if rays.size < 7:
        return 0.0, False, False, 0.0

    left = float(np.mean(rays[:3]))
    right = float(np.mean(rays[4:7]))
    forward = float(np.min(rays[2:5]))
    opportunity = (
        min(left, right) < RAY_DANGER_CLEARANCE
        and abs(left - right) >= RAY_IMBALANCE_MIN
    )
    if not opportunity:
        return 0.0, False, False, forward

    desired = STEER_RIGHT if left < right else STEER_LEFT
    toward = STEER_LEFT if desired == STEER_RIGHT else STEER_RIGHT
    if steer == desired:
        return 1.0, True, False, forward
    if steer == toward:
        return -1.0, True, True, forward
    return 0.0, True, False, forward


class FZeroEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        use_mmf=True,
        instance=None,
        frame_skip=DEFAULT_FRAME_SKIP,
        warmup_speed=0,
        hold_speed=0,
        safety_assist=True,
        assist_mode="full",
        curriculum_checkpoints=0,
        start_slots=None,
        episode_frame_limit=MAX_EPISODE_FRAMES,
        terminate_on_wall=True,
        terminate_on_stall=True,
        min_speed_floor=MIN_SPEED_FLOOR,
        min_speed_frames=MIN_SPEED_FRAMES_LIMIT,
        wall_contact_frames=WALL_CONTACT_FRAMES_LIMIT,
        turn_commitment=TURN_COMMITMENT_FRAMES,
        reward_mode="potential",
    ):
        super().__init__()

        self.frame_skip = max(int(frame_skip), 1)
        self.turn_commitment = max(int(turn_commitment), 1)

        # "potential" adds Ng-Harada-Russell shaping on track position and
        # drops the proxy terms it replaces; "legacy" is the hand-weighted
        # stack, kept so the two can be compared rather than assumed.
        if reward_mode not in ("potential", "legacy"):
            raise ValueError("reward_mode must be 'potential' or 'legacy'")
        self.reward_mode = reward_mode
        self._potential = 0.0

        # Hand control over already centred and up to speed, rather than
        # stationary on the start line. Every episode otherwise begins in a
        # situation the agent barely revisits, and the first thing it has to
        # learn is how to get out of it.
        self.warmup_speed = int(warmup_speed)

        # When set, the env works the throttle to hold this speed and the
        # agent's throttle choice is ignored. That leaves it learning
        # steering alone against a constant speed -- a much smaller problem,
        # and a base to build on before the throttle is handed back.
        self.hold_speed = int(hold_speed)
        self.curriculum_checkpoints = max(int(curriculum_checkpoints), 0)
        self.episode_frame_limit = max(int(episode_frame_limit), 1)
        self.terminate_on_wall = bool(terminate_on_wall)
        self.terminate_on_stall = bool(terminate_on_stall)
        # The machine must keep moving; see MIN_SPEED_FLOOR. 0 disables.
        self.min_speed_floor = max(int(min_speed_floor), 0)
        self.min_speed_frames = max(int(min_speed_frames), 0)

        # Per-env so the threshold can be A/B'd without editing the module
        # constant -- this project's rule is to re-measure rather than assume,
        # and a run cannot compare two values it has to edit source to change.
        # 0 disables contact-termination outright.
        self.wall_contact_frames = max(int(wall_contact_frames), 0)

        # Permanent guard rails.  Training and evaluation both go through the
        # environment, so genomes learn the residual driving problem around
        # exactly the same wall protection they will use when deployed.
        self._assist = DrivingAssist(enabled=safety_assist, mode=assist_mode)
        self.start_slots = tuple(int(slot) for slot in (start_slots or ()))
        if any(slot < 0 or slot > 9 for slot in self.start_slots):
            raise ValueError("savestate slots must be between 0 and 9")
        self._next_start_slot = 0
        self._current_start_slot = None
        self._current_start_variant = ()

        # Steer / throttle / shoulder as separate dimensions, so PPO can
        # learn each one without enumerating all 27 combinations.
        # DEFAULT, not an option: the agent has no brake at all.
        #
        # Steering is the whole job -- hold the throttle down and place the
        # machine across the road with left/right, nothing else.
        #
        # This removes every form of braking, and there are two, because the
        # game has no brake button. tools/find_buttons.py measured it: Y, A,
        # X, L and R all shed exactly 80 speed a frame, which is identical to
        # simply releasing the accelerator. So coasting is one brake and the
        # shoulder buttons are the other -- "lean" also scrubs speed. Dropping
        # the throttle choice alone would leave half of it in place.
        #
        # The action space collapses from MultiDiscrete([3, 2, 3]) -- 18
        # combinations -- to Discrete(3). That is a large secondary win:
        # exploration has three things to try instead of eighteen, and DQN
        # can use it directly without env/flat_action.py flattening anything.
        # Throttle is deliberately not configurable: coasting is the game's
        # only true brake. Lean is retained, though, because L/R are steering
        # technique as well as speed-shedding inputs. The policy therefore
        # chooses steering and lean, never throttle: 3 x 3 = 9 actions.
        self.action_space = spaces.MultiDiscrete((ACTION_DIMS[0], ACTION_DIMS[2]))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(NUM_FEATURES,), dtype=np.float32
        )

        self.bridge = BridgeClient(use_mmf=use_mmf, instance=instance)

        self._step_count = 0
        self._no_progress_count = 0
        self._slow_frames = 0
        self._start_lap = 0
        self._last_progress = 0
        self._distance = 0
        self._furthest_distance = 0
        self._laps_credited = 0
        self._lap_started_step = 0
        self._blind_steps = 0
        self._wall_steps = 0
        self._wall_contact_streak = 0
        self._reward_parts = dict.fromkeys(REWARD_PARTS, 0.0)
        self._last_speed = 0
        self._last_frame = None
        self._last_readings = []
        self._episode_start_progress = 0
        self._net_progress = 0
        self._assist_frames = 0
        self._safety_overrides = 0
        self._reversal_blocks = 0
        self._risk_total = 0.0
        self._centre_total = 0.0
        self._speed_total = 0.0
        self._speed_loss_total = 0.0
        self._ray_response_total = 0.0
        self._ray_opportunities = 0
        self._ray_toward_wall_frames = 0
        self._ray_clearance_recovery_total = 0.0
        self._last_ray_distances = np.zeros(7, dtype=np.float32)
        self._applied_steer_frames = [0, 0, 0]
        self._applied_shoulder_frames = [0, 0, 0]

    @property
    def last_frame(self):
        """The most recent emulator frame, or None if it could not be read."""
        return self._last_frame

    def _look(self, speed, frame):
        """Observation plus what the reward needs to know about the frame."""
        normalised = min(speed / MAX_SPEED_RAW, 1.0)

        if frame is None:
            assist = self._assist.observe([])
            self._last_readings = []
            return (
                features_from_readings([], normalised),
                {
                    "wall": False,
                    "road_pixels": 0,
                    "off_centre": 0.0,
                    "wall_risk": assist["risk"],
                    "assist_signal": assist["signal"],
                    "curve_signal": assist["curve_signal"],
                },
            )

        readings = measure_rows(frame, list(SCAN_ROWS))
        self._last_readings = readings
        assist = self._assist.observe(readings)
        rays = ray_distances(frame)
        observation = features_from_readings(readings, normalised, rays)
        return observation, {
            "wall": touching_wall(readings),
            "road_pixels": sum(r["road_pixels"] for r in readings),
            "off_centre": off_centre(readings),
            "wall_risk": assist["risk"],
            "assist_signal": assist["signal"],
            "curve_signal": assist["curve_signal"],
            "ray_distances": rays.tolist(),
        }

    def _observe(self, speed, frame):
        return self._look(speed, frame)[0]

    def _warm_up(self):
        """Drive to a centred, up-to-speed start before handing over.

        Uses the hand-written controller for steering only, so the machine
        arrives in the middle of the road rather than wherever the savestate
        left it. Gives up after a bounded number of frames so a bad start
        cannot hang the episode.
        """
        from control import SCAN_FRACTIONS, Controller

        controller = Controller()
        speed = 0
        frame = None

        for _ in range(WARMUP_MAX_FRAMES):
            speed, frame = self.bridge.advance(
                encode_action(*self._warmup_action(controller, frame, speed))
            )

            if frame is None:
                continue

            if speed >= self.warmup_speed and self._is_centred(frame):
                break

        return speed, frame

    def _warmup_action(self, controller, frame, speed):
        if frame is None:
            return STEER_STRAIGHT, THROTTLE_ACCEL, SHOULDER_NONE

        from control import SCAN_FRACTIONS

        height, width = frame.shape[:2]
        readings = measure_rows(frame, [int(height * f) for f in SCAN_FRACTIONS])
        steer, _, shoulder = controller.update(readings, width / 2.0, speed)[0]

        throttle = THROTTLE_ACCEL if speed < self.warmup_speed else THROTTLE_COAST
        return steer, throttle, shoulder

    def _is_centred(self, frame):
        """Roughly equal room either side on the row the machine sits on."""
        readings = measure_rows(frame, list(SCAN_ROWS))
        near = readings[-1]

        left = near["left_px"]
        right = near["right_px"]
        if left is None or right is None:
            # Road running off screen one way; nothing to centre against, so
            # accept it rather than warming up forever.
            return True

        total = left + right
        if total <= 0:
            return False

        return abs(right - left) / total <= WARMUP_CENTRE_TOLERANCE

    def _throttle_for(self, requested, speed):
        """Override the throttle when holding a constant speed."""
        if self.hold_speed <= 0:
            return requested

        return THROTTLE_ACCEL if speed < self.hold_speed else THROTTLE_COAST

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        options = options or {}
        requested_slot = options.get("start_slot")
        start_steers = tuple(int(steer) for steer in options.get("start_steers", ()))
        if len(start_steers) > 8:
            raise ValueError("start variant may advance at most 8 frames")
        if any(steer not in (STEER_STRAIGHT, STEER_LEFT, STEER_RIGHT)
               for steer in start_steers):
            raise ValueError("start variant steering must be straight, left, or right")
        if requested_slot is not None:
            slot = int(requested_slot)
            if not 0 <= slot <= 9:
                raise ValueError("savestate slot must be between 0 and 9")
            self._current_start_slot = slot
            speed, frame = self.bridge.reset(slot=slot)
        elif self.start_slots:
            slot = self.start_slots[self._next_start_slot % len(self.start_slots)]
            self._next_start_slot += 1
            self._current_start_slot = slot
            speed, frame = self.bridge.reset(slot=slot)
        else:
            self._current_start_slot = None
            speed, frame = self.bridge.reset()

        if self.warmup_speed > 0:
            speed, frame = self._warm_up()

        # Deterministic domain variation for evolutionary evaluation. Every
        # genome in a generation receives the same slot/variant schedule, so
        # comparisons stay fair. The prelude always holds acceleration and
        # never leans or coasts; it only creates a few nearby positions and
        # headings around each hand-built savestate. Episode accounting starts
        # afterwards, so these setup frames earn no progress or speed fitness.
        for steer in start_steers:
            speed, frame = self.bridge.advance(
                encode_action(steer, THROTTLE_ACCEL, SHOULDER_NONE)
            )
        self._current_start_variant = start_steers

        self._last_frame = frame

        self._assist.reset()
        self._last_readings = []

        self._step_count = 0
        self._no_progress_count = 0
        self._slow_frames = 0
        self._start_lap = self.bridge.lap
        self._last_progress = self.bridge.total_progress
        # Distance is measured FROM the spawn, so it starts at zero whatever
        # checkpoint the savestate happens to sit on. That is what makes every
        # spawn the same task.
        self._distance = 0
        self._furthest_distance = 0
        self._laps_credited = 0
        self._lap_started_step = 0
        self._blind_steps = 0
        self._wall_steps = 0
        self._wall_contact_streak = 0
        self._reward_parts = dict.fromkeys(REWARD_PARTS, 0.0)
        self._last_speed = speed
        self._episode_start_progress = self.bridge.total_progress

        # PHI is relative to the episode start, so it starts at zero -- and it
        # MUST be reset here. Leaving it set from the previous episode makes
        # the first shaping term of every episode a difference against
        # wherever the last one ended.
        self._potential = 0.0
        self._net_progress = 0
        self._assist_frames = 0
        self._safety_overrides = 0
        self._reversal_blocks = 0
        self._risk_total = 0.0
        self._centre_total = 0.0
        self._speed_total = 0.0
        self._speed_loss_total = 0.0
        self._ray_response_total = 0.0
        self._ray_opportunities = 0
        self._ray_toward_wall_frames = 0
        self._ray_clearance_recovery_total = 0.0
        self._applied_steer_frames = [0, 0, 0]
        self._applied_shoulder_frames = [0, 0, 0]

        observation, initial_seen = self._look(speed, frame)
        self._last_ray_distances = np.asarray(
            initial_seen.get("ray_distances", ()), dtype=np.float32
        )
        return observation, self._info(speed)

    def _info(self, speed):
        return {
            "speed": speed,
            "lap": self.bridge.lap,
            "progress": self.bridge.progress,
            # laps and distance both come from the odometer now, so they agree
            # with each other and do not depend on the spawn position.
            "laps_done": int(self._furthest_distance // CHECKPOINTS_PER_LAP),
            "furthest": self._furthest_distance,
            "start_progress": self._episode_start_progress,
            "progress_gain": max(self._furthest_distance, 0),
            # Signed displacement is the genetic time-trial objective. Unlike
            # progress_gain, reversing below the starting checkpoint makes it
            # negative instead of being clipped to zero.
            "net_progress": self._net_progress,
            "episode_frames": self._assist_frames,
            "start_slot": self._current_start_slot,
            "start_variant": list(self._current_start_variant),
            "wall_steps": self._wall_steps,
            "wall_contact_streak": self._wall_contact_streak,
            "racing": self.bridge.racing,
            "reward_parts": dict(self._reward_parts),
            "safety_overrides": self._safety_overrides,
            "safety_override_rate": self._safety_overrides / max(self._assist_frames, 1),
            "reversal_blocks": self._reversal_blocks,
            "mean_wall_risk": self._risk_total / max(self._assist_frames, 1),
            "mean_off_centre": self._centre_total / max(self._assist_frames, 1),
            "mean_speed_normalised": self._speed_total / max(self._assist_frames, 1),
            "speed_loss": self._speed_loss_total,
            "mean_speed_loss": self._speed_loss_total / max(self._assist_frames, 1),
            "ray_response_score": self._ray_response_total
            / max(self._ray_opportunities, 1),
            "ray_response_opportunities": self._ray_opportunities,
            "ray_toward_wall_rate": self._ray_toward_wall_frames
            / max(self._ray_opportunities, 1),
            "mean_ray_clearance_recovery": self._ray_clearance_recovery_total
            / max(self._ray_opportunities, 1),
            "actual_lean_rate": (
                self._applied_shoulder_frames[1] + self._applied_shoulder_frames[2]
            ) / max(self._assist_frames, 1),
            "applied_steer_frames": list(self._applied_steer_frames),
            "applied_shoulder_frames": list(self._applied_shoulder_frames),
        }

    def _progress_gain(self):
        """New ground covered since the last step, in checkpoints.

        Distance is UNWRAPPED, not read raw. Track position is a modulo-90
        counter, so crossing the start/finish line reads as a -89 step. The
        previous version treated that as a teleport, returned zero, and shifted
        the episode's baseline by the same amount -- which corrupted reporting,
        because that baseline is also what progress_gain subtracts. Measured on
        a live PPO run, start_progress went NEGATIVE (-52, -81) and
        progress_gain jumped ~89 in a single frame at every line crossing, so a
        "gain of 170" was really about 80.

        Unwrapping is safe here because track_map's search is windowed: it can
        only move a few checkpoints per lookup, so a jump of more than half a
        lap can only be the modulo boundary, never real motion.

        Measured against the FURTHEST distance reached, not the current one, so
        the reward is strictly for new ground. Paying for the raw change lets
        the agent farm reward by rocking back and forth over a boundary.
        """
        current = self.bridge.total_progress
        step = current - self._last_progress
        if abs(step) > PROGRESS_WRAP_THRESHOLD:
            # Two things produce a jump this large, and they need opposite
            # handling. A modulo wrap at the start/finish line is real motion
            # and must be unwrapped; a savestate reload or a map glitch is not
            # motion at all and must be ignored.
            #
            # They are separable because track_map's search is windowed: it
            # cannot move more than LOOK_AHEAD checkpoints in one lookup. So a
            # jump that becomes a plausible step once a lap is added or removed
            # was a wrap, and one that stays implausible either way is an
            # anomaly. Unwrapping unconditionally turns a +50 teleport into a
            # -40 "movement", which is just as impossible as the +50 was.
            unwrapped = (
                step + CHECKPOINTS_PER_LAP if step < 0 else step - CHECKPOINTS_PER_LAP
            )
            if abs(unwrapped) > MAX_PLAUSIBLE_STEP:
                self._last_progress = current
                return 0, 0
            step = unwrapped
        self._last_progress = current

        # Signed odometer from the spawn point, in checkpoints. Unbounded, so
        # it keeps counting across laps instead of freezing at 89.
        self._distance += step
        self._net_progress = self._distance
        lost = max(-step, 0)

        if self._distance <= self._furthest_distance:
            return 0, lost

        gain = self._distance - self._furthest_distance
        self._furthest_distance = self._distance
        return gain, lost

    def step(self, action):
        """Hold the action for frame_skip emulated frames.

        The machine does not respond meaningfully within a sixtieth of a
        second, so deciding that often mostly burns experience: at one frame
        per step a whole hour of training is only about 90 laps. Holding an
        action for several frames buys proportionally more track per step
        and shortens the horizon the policy has to reason over.

        Reward accumulates across the held frames, so a step is worth about
        frame_skip times what it was.
        """
        # Throttle is pinned on: there is no coast/brake action. The two
        # remaining choices are steering and the L/R lean button.
        steer, shoulder = (int(value) for value in action)
        throttle = THROTTLE_ACCEL

        reward = 0.0
        observation = None
        terminated = False
        truncated = False
        info = {}
        step_override_frames = 0
        step_reversal_blocks = 0
        step_reasons = []
        applied_actions = []

        # A turn is committed to for a minimum number of frames, because a
        # two-frame steering input barely moves the machine (see
        # TURN_COMMITMENT_FRAMES). Straight-ahead steps keep the normal skip.
        action_budget = self._assist.action_hold(self.frame_skip)
        if steer != STEER_STRAIGHT:
            action_budget = max(action_budget, self.turn_commitment)

        for _ in range(action_budget):
            (applied_steer, applied_shoulder), assist = self._assist.apply(
                steer, shoulder
            )
            applied_actions.append((applied_steer, applied_shoulder))
            self._applied_steer_frames[applied_steer] += 1
            self._applied_shoulder_frames[applied_shoulder] += 1
            if assist["override"]:
                self._safety_overrides += 1
                step_override_frames += 1
                step_reasons.append(assist["reason"])
            if assist["reversal_blocked"]:
                self._reversal_blocks += 1
                step_reversal_blocks += 1

            # Re-encoded every frame because holding a constant speed needs
            # the throttle decided against the CURRENT speed, not the speed
            # when the step began.
            code = encode_action(
                applied_steer,
                self._throttle_for(throttle, self._last_speed),
                applied_shoulder,
            )
            observation, step_reward, terminated, truncated, info = self._one_frame(
                code
            )
            reward += step_reward
            if terminated or truncated:
                break

            # If a wall or sharp curve appeared during a normally long hold,
            # return control to the policy immediately instead of blindly
            # repeating the stale action for all four frames.
            #
            # A committed turn is exempt UNLESS the assist calls an emergency:
            # otherwise the commitment is undone on its second frame by the
            # very curve that justified turning, which is the behaviour this
            # is meant to fix. The action is still re-corrected every frame by
            # _assist.apply above, so a held turn is never a stale one.
            allowed = self._assist.action_hold(self.frame_skip)
            if steer != STEER_STRAIGHT and not self._assist.state["emergency"]:
                allowed = max(allowed, self.turn_commitment)
            if len(applied_actions) >= allowed:
                break

        info["safety_override"] = step_override_frames > 0
        info["safety_override_frames"] = step_override_frames
        info["reversal_blocked_frames"] = step_reversal_blocks
        info["safety_reason"] = step_reasons[-1] if step_reasons else "policy"
        info["frames_in_step"] = len(applied_actions)
        info["applied_action"] = applied_actions[-1] if applied_actions else (steer, shoulder)
        info["applied_actions"] = list(applied_actions)

        return observation, reward, terminated, truncated, info

    def _phi(self):
        """Potential of the CURRENT state: distance driven from the spawn.

        A potential has to be a function of the state alone for Ng's invariance
        to hold, and using distance is what makes rocking back and forth over a
        checkpoint boundary worth exactly nothing -- consecutive terms
        telescope.

        This reads the unwrapped odometer, which is already relative to where
        the episode began and already immune to the modulo wrap. It used to
        subtract a mutable baseline that _progress_gain edited on every wrap,
        which is what let a reporting bug and a shaping fix corrupt each other.
        Absolute position cannot be used: it keeps growing across laps, so
        PHI(terminal) = 0 would charge an ever larger exit toll and made
        finishing two laps score -3727 against -33 for giving up.
        """
        return POTENTIAL_SCALE * float(self._distance)

    def _one_frame(self, code):
        previous_speed = self._last_speed
        applied_steer = decode_action(code)[0]
        response, ray_opportunity, toward_wall, previous_clearance = ray_response(
            self._last_ray_distances, applied_steer
        )
        speed, frame = self.bridge.advance(code)
        self._last_speed = speed
        speed_loss = max(previous_speed - speed, 0)
        speed_loss_scale = (
            DANGER_SPEED_LOSS_MULTIPLIER if ray_opportunity else 1.0
        )
        weighted_speed_loss = speed_loss * speed_loss_scale
        self._speed_loss_total += weighted_speed_loss

        # Kept so a teacher can steer from the picture while the env stays in
        # charge of rewards, resets and recording (training/record_teacher.py).
        # The hand-written controller scans its own rows at its own heights,
        # which the compact row-and-ray observation cannot be reconstructed from.
        self._last_frame = frame

        observation, seen = self._look(speed, frame)
        current_rays = np.asarray(seen.get("ray_distances", ()), dtype=np.float32)
        _, _, _, current_clearance = ray_response(current_rays, applied_steer)
        clearance_gain = (
            max(current_clearance - previous_clearance, 0.0)
            if ray_opportunity
            else 0.0
        )
        if ray_opportunity:
            self._ray_opportunities += 1
            self._ray_response_total += response
            self._ray_toward_wall_frames += int(toward_wall)
            self._ray_clearance_recovery_total += clearance_gain
        self._last_ray_distances = current_rays
        self._assist_frames += 1
        self._risk_total += float(seen.get("wall_risk", 0.0))
        self._centre_total += float(seen.get("off_centre", 0.0))
        self._speed_total += min(max(float(speed) / TOP_SPEED_RAW, 0.0), 1.0)
        gain, lost = self._progress_gain()

        # Tracked separately so a run can be broken down into where the
        # reward actually came from -- a flat learning curve is usually one
        # term drowning out the others.
        parts = self._reward_parts
        progress_reward = gain * (REWARD_PER_CHECKPOINT + speed * REWARD_SPEED_SCALE)
        speed_reward = (speed - SPEED_REWARD_BASELINE) * REWARD_SPEED_PER_STEP
        speed_loss_penalty = -weighted_speed_loss * PENALTY_SPEED_DECREASE_PER_UNIT
        ray_response_penalty = -PENALTY_STEER_TOWARD_WALL if toward_wall else 0.0
        ray_clearance_reward = clearance_gain * REWARD_RAY_CLEARANCE_GAIN
        lost_penalty = -lost * PENALTY_PER_CHECKPOINT_LOST

        # Bounded above by zero, like the speed term and for the same
        # reason: a positive per-step payout is banked more times by a slow
        # lap than a fast one.
        centre_penalty = -seen.get("off_centre", 0.0) * PENALTY_OFF_CENTRE

        parts["progress"] += progress_reward
        parts["speed"] += speed_reward
        parts["speed_loss"] += speed_loss_penalty
        parts["ray_response"] += ray_response_penalty
        parts["ray_clearance"] += ray_clearance_reward
        parts["centre"] += centre_penalty
        parts["lost"] += lost_penalty
        parts["step_cost"] -= REWARD_STEP_COST

        if self.reward_mode == "potential":
            # The dropped terms are proxies the potential now signals
            # directly -- see POTENTIAL_MODE_DROPS. They are still recorded in
            # `parts` above so the dashboard can show what they WOULD have
            # charged, but they no longer reach the learner.
            for dropped in POTENTIAL_MODE_DROPS:
                parts[dropped] = 0.0
            reward = (
                progress_reward
                + speed_reward
                + speed_loss_penalty
                + ray_clearance_reward
                + lost_penalty
            )
        else:
            reward = (
                progress_reward
                + speed_reward
                + speed_loss_penalty
                + ray_response_penalty
                + ray_clearance_reward
                + centre_penalty
                + lost_penalty
                - REWARD_STEP_COST
            )

        if seen["wall"]:
            # Charged every frame of contact. Scraping is what drains power
            # and eventually destroys the machine, and it is also the only
            # way the per-step speed reward could be farmed while going
            # nowhere.
            self._wall_steps += 1
            reward -= PENALTY_WALL_CONTACT
            parts["wall"] -= PENALTY_WALL_CONTACT
            self._wall_contact_streak += 1
        else:
            self._wall_contact_streak = 0

        self._step_count += 1
        if gain > 0:
            self._no_progress_count = 0
        else:
            self._no_progress_count += 1

        # The race being over is the game's own answer, not a guess from the
        # picture. This fires on the results screen, the course card and the
        # menus alike, within a few frames rather than the 30 the old
        # road-pixel test needed to confirm.
        if self.bridge.racing:
            self._blind_steps = 0
        else:
            self._blind_steps += 1

        crashed = self._blind_steps >= NOT_RACING_CONFIRM_FRAMES
        if crashed:
            reward -= PENALTY_CRASH
            parts["crash"] -= PENALTY_CRASH

        # A lap is DISTANCE DRIVEN, not the start/finish line going past.
        #
        # The line-crossing test made the task cheaper the further along the
        # machine spawned, because it only ever had to cover the remainder.
        # Measured on a live PPO run against a recorded human lap, episodes
        # flagged "finished" had actually covered:
        #
        #   spawn slot 2 (checkpoint  9)   60% of a lap
        #   spawn slot 4 (checkpoint 12)   56%
        #   spawn slot 5 (checkpoint 29)   43%
        #   spawn slot 6 (checkpoint 41)   33%
        #
        # A genuine lap covers the same ground wherever it starts -- it is one
        # closed loop -- so that gradient is the giveaway. Counting distance
        # instead makes every spawn the same task, which is also what keeps
        # varied spawns from quietly biasing the comparison between learners.
        laps_done = int(self._furthest_distance // CHECKPOINTS_PER_LAP)
        newly_completed = laps_done - self._laps_credited
        finished = laps_done >= LAPS_TO_FINISH
        for _ in range(max(newly_completed, 0)):
            # Timed per lap, not per episode, so a second lap is judged on its
            # own pace rather than inheriting the first one's.
            lap_frames = max(self._step_count - self._lap_started_step, 1)
            speed_bonus = min(
                max(REFERENCE_LAP_FRAMES / lap_frames, LAP_SPEED_BONUS_MIN),
                LAP_SPEED_BONUS_MAX,
            )
            reward += REWARD_LAP_COMPLETE * speed_bonus
            parts["lap"] += REWARD_LAP_COMPLETE * speed_bonus
            self._lap_started_step = self._step_count
            self._laps_credited += 1

        curriculum_complete = (
            self.curriculum_checkpoints > 0
            and self._furthest_distance >= self.curriculum_checkpoints
        )

        # Ignore a brief scrape or a bad road-mask sample; treat only
        # sustained unbroken contact as a hit. The per-frame penalty still
        # applies from the first detected contact above, so a brush is
        # costly even when it is survivable.
        #
        # A threshold of 0 means contact never terminates on its own -- the
        # no-progress timer and the game's own destruction flag still do.
        wall_hit = (
            self.wall_contact_frames > 0
            and self._wall_contact_streak >= self.wall_contact_frames
        )

        # The machine must keep MOVING. Acceleration is always held, so being
        # slow means scrubbing against something -- and in time-trial mode
        # wall contact does not end the episode, so a wedged genome would
        # otherwise grind out the whole trial for free.
        if self.min_speed_floor and speed < self.min_speed_floor:
            self._slow_frames += 1
        else:
            self._slow_frames = 0
        stalled_out = bool(
            self.min_speed_floor
            and self.min_speed_frames
            and self._slow_frames >= self.min_speed_frames
        )

        terminated = (
            finished
            or curriculum_complete
            or crashed
            or (wall_hit and self.terminate_on_wall)
            or (
                self.terminate_on_stall
                and self._no_progress_count >= NO_PROGRESS_FRAMES_LIMIT
            )
            or stalled_out
        )
        # Limits count emulated frames, never policy steps. action_hold can
        # vary from one to four frames near a wall, so _step_count would give
        # different candidates different amounts of game time.
        truncated = self._assist_frames >= self.episode_frame_limit

        if self.reward_mode == "potential":
            # F = gamma * PHI(s') - PHI(s), with PHI(terminal) = 0.
            #
            # Zeroing at termination is not a detail, it is the whole point:
            # it is what makes ending an episode cost the position you had
            # reached, which is the terminal penalty the legacy reward was
            # missing. Truncation is NOT terminal -- the episode was cut by a
            # clock, not by failing -- so the potential carries over there and
            # the learner bootstraps as usual.
            next_potential = 0.0 if terminated else self._phi()
            shaping = POTENTIAL_GAMMA * next_potential - self._potential
            self._potential = next_potential
            reward += shaping
            parts["potential"] += shaping

        info = self._info(speed)
        info["finished"] = finished
        info["crashed"] = crashed
        info["stalled_out"] = stalled_out
        info["slow_frames"] = self._slow_frames
        info["curriculum_complete"] = curriculum_complete
        info["wall"] = seen["wall"]
        info["wall_hit"] = wall_hit
        info["wall_contact_streak"] = self._wall_contact_streak
        # Exposed to whole-episode genetic fitness. Zero is perfectly
        # centred and one is against a visible wall.
        info["off_centre"] = seen.get("off_centre", 0.0)
        info["wall_risk"] = seen.get("wall_risk", 0.0)
        info["assist_signal"] = seen.get("assist_signal", 0.0)
        info["curve_signal"] = seen.get("curve_signal", 0.0)
        # Keep the geometric observation available to run recording and the
        # dashboard as well as to the policy.  Omitting this copy made live
        # seven-ray plots read as all zero even though the policy input was
        # correct.
        info["ray_distances"] = list(seen.get("ray_distances", ()))
        return observation, reward, terminated, truncated, info

    def close(self):
        self.bridge.close()
