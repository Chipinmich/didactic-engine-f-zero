"""Live view of a training run, in a browser.

Reads the run store (training/run_store.py) and redraws every few seconds, so
a long run can be watched while it happens rather than judged from a number
printed at the end. No run has yet gone far enough to show whether the policy
learns at all, and the useful thing to know early is which way the curve is
pointing.

    python tools/dashboard.py                 newest run, http://localhost:8321
    python tools/dashboard.py --run teacher_20260820_193000
    python tools/dashboard.py --port 9000

Charts are inline SVG drawn here: no javascript library to fetch, nothing to
install, and it works with the machine offline.

What to look at, in order:

  furthest      the one that matters. Reward can climb while the machine
                gets no further, because the speed term pays every step.

  wall contact  the measured bottleneck -- about 30% of frames touch a wall,
                which costs 2.8% of power per checkpoint and kills the run
                by checkpoint 35.

  reward parts  a flat learning curve is usually one term drowning out the
                others. This is the quickest way to see that.
"""

import argparse
import html
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from training.genetic import rank_key
from training.run_store import list_runs, load_episodes, load_generations, load_meta

REFRESH_SECONDS = 5

# Episodes are noisy enough here that the raw series says very little; the
# trend is the point. Kept explicit so nobody reads a smoothed line as data.
SMOOTH_WINDOW = 20
DEFAULT_ASSESS_WINDOW = 100
MIN_ASSESS_WINDOW = 10
MAX_ASSESS_WINDOW = 2000

WIDTH = 820
HEIGHT = 200
PAD_LEFT = 62
PAD_BOTTOM = 28
PAD_TOP = 14

STYLE = """
:root { color-scheme: dark; }
body { background:#12141a; color:#d8dae0; font:14px ui-monospace,Consolas,monospace;
       margin:0; padding:24px 28px; }
h1 { font-size:17px; margin:0 0 2px; color:#fff; font-weight:600; }
h2 { font-size:13px; margin:26px 0 6px; color:#9aa0ac; font-weight:600;
     text-transform:uppercase; letter-spacing:.06em; }
.sub { color:#767c88; margin-bottom:18px; }
.cards { display:flex; flex-wrap:wrap; gap:10px; margin:14px 0 4px; }
.card { background:#1a1d25; border:1px solid #262a33; border-radius:6px;
        padding:9px 14px; min-width:118px; }
.card .k { color:#767c88; font-size:11px; text-transform:uppercase;
           letter-spacing:.05em; }
.card .v { font-size:19px; color:#fff; margin-top:3px; }
svg { background:#1a1d25; border:1px solid #262a33; border-radius:6px;
      display:block; max-width:100%; }
a { color:#6ea8fe; text-decoration:none; margin-right:14px; }
a.on { color:#fff; font-weight:600; }
table { border-collapse:collapse; margin-top:6px; }
td,th { padding:3px 14px 3px 0; text-align:right; }
th { color:#767c88; font-weight:600; }
td.l,th.l { text-align:left; }
.note { color:#767c88; margin-top:8px; max-width:820px; line-height:1.5; }
.assessment { max-width:820px; background:#1a1d25; border:1px solid #262a33;
              border-radius:6px; padding:14px 16px; }
.verdict { font-size:18px; font-weight:600; margin-bottom:4px; }
.verdict.good, .direction.good { color:#5fd3a6; }
.verdict.bad, .direction.bad { color:#ff7b72; }
.verdict.mixed, .direction.mixed { color:#d29922; }
.direction { font-weight:600; text-align:left; }
form.window { margin:8px 0 12px; color:#9aa0ac; }
input[type=number] { width:72px; background:#12141a; color:#fff;
                     border:1px solid #3d434f; border-radius:4px; padding:4px 6px; }
button { background:#262a33; color:#d8dae0; border:1px solid #3d434f;
         border-radius:4px; padding:4px 9px; cursor:pointer; }
"""

# Colour-blind safe, and each series keeps its colour across every chart.
COLOURS = {
    "reward": "#6ea8fe",
    "furthest": "#5fd3a6",
    "wall": "#f0883e",
    "speed": "#c297ff",
    "centre": "#5fd3a6",
    "risk": "#ff7b72",
    "assist": "#d29922",
    "lean": "#58a6ff",
    "requested": "#c297ff",
    "turn": "#7ee787",
    "agreement": "#79c0ff",
    "rays": "#ffa657",
    "ray_response": "#39d353",
    "toward_wall": "#ff7b72",
}
PART_COLOURS = {
    "progress": "#5fd3a6",
    "speed": "#6ea8fe",
    "speed_loss": "#c297ff",
    "ray_response": "#ff7b72",
    "ray_clearance": "#39d353",
    "wall": "#f0883e",
    "crash": "#ff7b72",
    "lost": "#d29922",
    "step_cost": "#8b949e",
    "lap": "#ffd866",
}


def smooth(values, window=SMOOTH_WINDOW):
    if len(values) < 2:
        return list(values)

    window = max(1, min(window, len(values)))
    out = []
    total = 0.0
    for i, value in enumerate(values):
        total += value
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def line_chart(values, colour, label, fmt="{:.0f}", baseline_zero=False):
    """One series against episode number, with a smoothed line over it."""
    if not values:
        return f'<p class="sub">no {html.escape(label)} yet</p>'

    low = min(values)
    high = max(values)
    if baseline_zero:
        low = min(low, 0.0)
    if high == low:
        high = low + 1.0

    span = high - low
    plot_w = WIDTH - PAD_LEFT - 12
    plot_h = HEIGHT - PAD_TOP - PAD_BOTTOM

    def x(i):
        return PAD_LEFT + (plot_w * i / max(len(values) - 1, 1))

    def y(v):
        return PAD_TOP + plot_h - (plot_h * (v - low) / span)

    raw = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(values))
    trend = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(smooth(values)))

    # Three gridlines: bottom, middle, top.
    grid = []
    for fraction in (0.0, 0.5, 1.0):
        value = low + span * fraction
        gy = y(value)
        grid.append(
            f'<line x1="{PAD_LEFT}" y1="{gy:.1f}" x2="{PAD_LEFT + plot_w}" '
            f'y2="{gy:.1f}" stroke="#262a33"/>'
            f'<text x="{PAD_LEFT - 8}" y="{gy + 4:.1f}" fill="#767c88" '
            f'font-size="11" text-anchor="end">{fmt.format(value)}</text>'
        )

    if baseline_zero and low < 0 < high:
        zy = y(0.0)
        grid.append(
            f'<line x1="{PAD_LEFT}" y1="{zy:.1f}" x2="{PAD_LEFT + plot_w}" '
            f'y2="{zy:.1f}" stroke="#3d434f" stroke-dasharray="3 3"/>'
        )

    return (
        f'<svg viewBox="0 0 {WIDTH} {HEIGHT}" width="{WIDTH}" height="{HEIGHT}" '
        f'role="img" aria-label="{html.escape(label)}">'
        + "".join(grid)
        + f'<polyline points="{raw}" fill="none" stroke="{colour}" '
        f'stroke-width="1" opacity="0.30"/>'
        + f'<polyline points="{trend}" fill="none" stroke="{colour}" '
        f'stroke-width="2"/>'
        + f'<text x="{PAD_LEFT}" y="{HEIGHT - 8}" fill="#767c88" font-size="11">'
        f'episode 1</text>'
        + f'<text x="{PAD_LEFT + plot_w}" y="{HEIGHT - 8}" fill="#767c88" '
        f'font-size="11" text-anchor="end">episode {len(values)}</text>'
        + "</svg>"
    )


def card(key, value):
    return (
        f'<div class="card"><div class="k">{html.escape(key)}</div>'
        f'<div class="v">{html.escape(str(value))}</div></div>'
    )


def verified_laps(episodes):
    """Only full-distance laps, never a finish-line/reporting false positive."""
    laps = []
    for episode in episodes:
        distance = episode.get(
            "progress_gain", episode.get("furthest", 0)
        )
        if (
            episode.get("finished")
            and episode.get("laps", 0) >= 1
            and episode.get("furthest", 0) >= 90
            and distance >= 90
            and episode.get("episode_frames", 0) > 0
        ):
            laps.append(episode)
    return laps


def timed_laps(episodes):
    """Verified laps eligible for an official start-to-finish time.

    Mixed-start training deliberately includes slot 7 near checkpoint 40.  It
    is useful evidence that the policy can finish from there, but its episode
    frame count is not comparable with a full lap from slot 0.  Legacy runs
    have no start-slot label, so retain their old leaderboard behaviour.
    """
    laps = verified_laps(episodes)
    if any(episode.get("start_slot") is not None for episode in episodes):
        return [episode for episode in laps if episode.get("start_slot") == 0]
    return laps


def lap_leaderboard(episodes, limit=10):
    """Fastest verified full laps from the run, in emulated-frame time."""
    laps = sorted(timed_laps(episodes), key=lambda e: e["episode_frames"])
    if not laps:
        return (
            "<h2>fastest verified laps</h2>"
            '<p class="sub">No full 90-checkpoint lap has been verified yet.</p>'
        )

    rows = []
    for rank, episode in enumerate(laps[: max(int(limit), 1)], 1):
        frames = int(episode["episode_frames"])
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{frames / 60.0:.2f}s</td>"
            f"<td>{frames}</td>"
            f"<td>{float(episode.get('mean_speed', 0.0)):.0f}</td>"
            f"<td>{float(episode.get('wall_rate', 0.0)) * 100:.1f}%</td>"
            f"<td>{html.escape(str(episode.get('start_slot', 'default')))}</td>"
            f"<td>{html.escape(str(episode.get('instance', '?')))}</td>"
            f"<td>{html.escape(str(episode.get('episode', '?')))}</td>"
            "</tr>"
        )
    return (
        "<h2>fastest verified laps</h2>"
        "<table><tr><th>rank</th><th>time</th><th>frames</th>"
        "<th>mean speed</th><th>wall</th><th>start</th>"
        "<th>instance</th><th>episode</th></tr>"
        + "".join(rows)
        + "</table>"
        + '<p class="note">A lap appears here only after covering all 90 '
        "checkpoints from its spawn. In mixed-start runs, only slot 0 laps "
        "are timed. Times use emulated frames, so host "
        "slow-motion or turbo speed cannot improve them.</p>"
    )


def _mean(episodes, getter):
    return sum(float(getter(episode)) for episode in episodes) / len(episodes)


def improvement_panel(episodes, window=DEFAULT_ASSESS_WINDOW):
    """Compare the newest episode window with the immediately preceding one."""
    window = max(MIN_ASSESS_WINDOW, min(int(window), MAX_ASSESS_WINDOW))
    needed = window * 2
    if len(episodes) < needed:
        return (
            "<h2>recent improvement assessment</h2>"
            '<div class="assessment"><div class="verdict mixed">Collecting evidence</div>'
            f'<div class="note">Need {needed} completed episodes to compare two '
            f'groups of {window}; {len(episodes)} are available.</div></div>'
        )

    previous = episodes[-needed:-window]
    recent = episodes[-window:]
    progress = lambda e: e.get(
        "net_progress", e.get("progress_gain", e.get("furthest", 0.0))
    )
    metrics = [
        ("signed progress", progress, "{:.2f}", 0.10, True, 3, "relative"),
        ("reward", lambda e: e.get("reward", 0.0), "{:+.1f}", 0.10, True, 1, "relative"),
        ("wall contact", lambda e: e.get("wall_rate", 0.0) * 100,
         "{:.1f}%", 2.0, False, 1, "absolute"),
        ("safety intervention", lambda e: e.get("safety_override_rate", 0.0) * 100,
         "{:.1f}%", 1.5, False, 1, "absolute"),
        ("turn quality", lambda e: e.get("turn_quality", 0.0) * 100,
         "{:.1f}%", 2.0, True, 1, "absolute"),
        ("steering toward wall", lambda e: e.get("ray_toward_wall_rate", 0.0) * 100,
         "{:.1f}%", 3.0, False, 1, "absolute"),
    ]

    rows = []
    score = 0
    for name, getter, fmt, threshold, higher_better, weight, threshold_kind in metrics:
        before = _mean(previous, getter)
        after = _mean(recent, getter)
        delta = after - before
        meaningful = (
            abs(delta) / max(abs(before), 1.0) >= threshold
            if threshold_kind == "relative" else abs(delta) >= threshold
        )
        favourable = delta > 0 if higher_better else delta < 0
        if meaningful:
            direction = "improving" if favourable else "worsening"
            css = "good" if favourable else "bad"
            score += weight if favourable else -weight
        else:
            direction = "roughly flat"
            css = "mixed"
        delta_fmt = "{:+.2f}" if name == "signed progress" else "{:+.1f}"
        delta_text = delta_fmt.format(delta) + (" pp" if "%" in fmt else "")
        rows.append(
            "<tr>"
            f'<td class="l">{html.escape(name)}</td>'
            f"<td>{fmt.format(before)}</td><td>{fmt.format(after)}</td>"
            f"<td>{delta_text}</td>"
            f'<td class="direction {css}">{direction}</td></tr>'
        )

    recent_laps = sum(1 for episode in recent if episode.get("finished"))
    previous_laps = sum(1 for episode in previous if episode.get("finished"))
    if recent_laps > previous_laps:
        score += 4
    elif recent_laps < previous_laps:
        score -= 4

    if score >= 3:
        verdict, css = "Going in the right direction", "good"
    elif score <= -3:
        verdict, css = "Trending in the wrong direction", "bad"
    else:
        verdict, css = "Mixed or inconclusive", "mixed"

    return (
        "<h2>recent improvement assessment</h2>"
        f'<div class="assessment"><div class="verdict {css}">{verdict}</div>'
        f'<div class="note">Latest {window} completed episodes compared with '
        f'the preceding {window}. Laps: {previous_laps} → {recent_laps}.</div>'
        '<table><tr><th class="l">signal</th><th>previous</th><th>latest</th>'
        '<th>change</th><th class="l">assessment</th></tr>'
        + "".join(rows)
        + '</table><p class="note">Progress and completed laps carry the most '
          'weight. Small changes are labelled flat so normal episode noise does '
          'not produce a confident verdict.</p></div>'
    )


def reward_parts_table(episodes):
    """Where the reward came from, over the last stretch of the run."""
    recent = episodes[-SMOOTH_WINDOW:]
    totals = {}
    for episode in recent:
        for key, value in (episode.get("reward_parts") or {}).items():
            totals[key] = totals.get(key, 0.0) + value

    if not totals:
        return '<p class="sub">no reward breakdown recorded</p>'

    scale = sum(abs(v) for v in totals.values()) or 1.0
    rows = sorted(totals.items(), key=lambda kv: -abs(kv[1]))

    body = "".join(
        f'<tr><td class="l" style="color:{PART_COLOURS.get(key, "#d8dae0")}">'
        f"{html.escape(key)}</td>"
        f"<td>{value / len(recent):+.1f}</td>"
        f"<td>{abs(value) / scale * 100:.0f}%</td></tr>"
        for key, value in rows
    )
    return (
        f'<table><tr><th class="l">term</th><th>per episode</th>'
        f"<th>share</th></tr>{body}</table>"
        f'<p class="note">Averaged over the last {len(recent)} episodes. '
        f"One term at 80% is the usual reason a curve looks flat.</p>"
    )


def lineage_panel(generations):
    """Completed-generation ranks, ancestry and parent/child improvement."""
    if not generations:
        return (
            "<h2>evolution and lineage</h2>"
            '<p class="note">No completed generation summary was recorded by '
            "this older run. Episode charts alone cannot prove which genomes "
            "were selected.</p>"
        )

    rows = []
    previous = {}
    for generation in generations:
        genomes = generation.get("genomes", [])
        ranked = sorted(genomes, key=lambda item: item.get("rank", 10**9))
        if not ranked:
            continue
        champion = ranked[0]
        score = champion.get("score", {})
        source_counts = {}
        improved = compared = 0
        for genome in genomes:
            source = genome.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            parents = [previous[parent] for parent in genome.get("parent_ids", [])
                       if parent in previous]
            if parents and source in {
                "bred", "broad_bred", "targeted_bred", "elite_variant"
            }:
                compared += 1
                if rank_key(genome.get("score", {})) > max(
                    rank_key(parent) for parent in parents
                ):
                    improved += 1
        source_text = " ".join(
            f"{html.escape(name)}:{count}"
            for name, count in sorted(source_counts.items())
        )
        child_text = f"{improved}/{compared}" if compared else "—"
        rows.append(
            "<tr>"
            f"<td>{generation.get('generation', '?')}</td>"
            f"<td class='l'>{html.escape(champion.get('genome_id', '?'))}</td>"
            f"<td>{html.escape(champion.get('source', 'unknown'))}</td>"
            f"<td>{score.get('distance', 0.0):+.1f}</td>"
            f"<td>{score.get('progress_score', 0.0):+.1f}</td>"
            f"<td>{score.get('wall_rate', 0.0) * 100:.1f}%</td>"
            f"<td>{score.get('speed', 0.0) * 100:.1f}%</td>"
            f"<td>{score.get('ray_response', 0.0) * 100:+.1f}%</td>"
            f"<td>{score.get('turn_quality', 0.0) * 100:.1f}%</td>"
            f"<td>{score.get('centre', 0.0) * 100:.1f}%</td>"
            f"<td>{child_text}</td>"
            f"<td class='l'>{source_text}</td>"
            "</tr>"
        )
        previous = {
            genome.get("genome_id"): genome.get("score", {}) for genome in genomes
        }

    return (
        "<h2>evolution and lineage</h2>"
        "<table><tr><th>gen</th><th class='l'>champion</th><th>origin</th>"
        "<th>distance</th><th>adjusted score</th><th>wall</th><th>speed</th>"
        "<th>ray response</th><th>turn</th><th>centre</th>"
        "<th>children &gt; parents</th><th class='l'>population sources</th></tr>"
        + "".join(rows)
        + "</table>"
        + '<p class="note">“Children &gt; parents” compares each bred child '
          "with its recorded parents under the exact GA rank: completed lap, "
          "whole signed checkpoint, then combined driving quality. The adjusted "
          "score is shown for diagnosis, not mistaken for the full rank. Only "
          "completed generations appear, so short crashes cannot bias this table.</p>"
    )


def render(run_id, runs, assess_window=DEFAULT_ASSESS_WINDOW):
    episodes = load_episodes(run_id)
    generation_records = load_generations(run_id)
    meta = load_meta(run_id)

    links = " ".join(
        f'<a href="/?{html.escape(urlencode({"run": name, "window": assess_window}))}" '
        f'class="{"on" if name == run_id else ""}">{html.escape(name)}</a>'
        for name in runs[:12]
    )

    head = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta http-equiv='refresh' content='{REFRESH_SECONDS}'>"
        f"<title>{html.escape(run_id)}</title><style>{STYLE}</style></head><body>"
        f"<h1>{html.escape(run_id)}</h1>"
        f"<div class='sub'>{html.escape(meta.get('source', 'training run'))} &middot; "
        f"refreshes every {REFRESH_SECONDS}s &middot; "
        f"{time.strftime('%H:%M:%S')}</div>"
        f"<div>{links}</div>"
        f'<form class="window" method="get"><input type="hidden" name="run" '
        f'value="{html.escape(run_id)}">Compare the latest '
        f'<input type="number" name="window" min="{MIN_ASSESS_WINDOW}" '
        f'max="{MAX_ASSESS_WINDOW}" value="{assess_window}"> episodes '
        '<button type="submit">Update assessment</button></form>'
    )

    if not episodes:
        return (
            head
            + "<h2>no episodes yet</h2>"
            + "<p class='sub'>Nothing has finished. An episode is written when "
            "it ends, so the first one appears after the first crash or lap.</p>"
            "</body></html>"
        )

    reward = [e.get("reward", 0.0) for e in episodes]
    furthest = [e.get("furthest", 0) for e in episodes]
    has_signed_distance = any("net_progress" in e for e in episodes)
    progress_gain = [
        e.get("net_progress", e.get("progress_gain", e.get("furthest", 0)))
        for e in episodes
    ]
    uses_turn_slots = any(e.get("start_slot") is not None for e in episodes)
    progress_label = (
        "signed distance" if has_signed_distance
        else "progress gained" if uses_turn_slots
        else "furthest checkpoint"
    )
    wall = [e.get("wall_rate", 0.0) * 100 for e in episodes]
    speed = [e.get("mean_speed", 0.0) for e in episodes]
    centre = [e.get("centre_quality", 0.0) * 100 for e in episodes]
    risk = [e.get("mean_wall_risk", 0.0) * 100 for e in episodes]
    assist = [e.get("safety_override_rate", 0.0) * 100 for e in episodes]
    actual_lean = [e.get("actual_lean_rate", 0.0) * 100 for e in episodes]
    requested_lean = [e.get("requested_lean_rate", 0.0) * 100 for e in episodes]
    frame_hold = [e.get("mean_frames_per_step", 0.0) for e in episodes]
    turn_quality = [e.get("turn_quality", 0.0) * 100 for e in episodes]
    policy_agreement = [e.get("policy_agreement", 0.0) * 100 for e in episodes]
    ray_clearance = [e.get("closest_ray_clearance", 0.0) * 100 for e in episodes]
    ray_response = [e.get("ray_response_score", 0.0) * 100 for e in episodes]
    toward_wall = [e.get("ray_toward_wall_rate", 0.0) * 100 for e in episodes]
    clearance_recovery = [
        e.get("mean_ray_clearance_recovery", 0.0) * 100 for e in episodes
    ]
    speed_loss = [e.get("mean_speed_loss", 0.0) for e in episodes]

    recent = episodes[-SMOOTH_WINDOW:]
    full_laps = verified_laps(episodes)
    official_laps = timed_laps(episodes)
    lap_window = episodes[-assess_window:]
    recent_full_laps = verified_laps(lap_window)
    instances = len({e.get("instance") for e in episodes})
    episode_generations = [int(e["generation"]) for e in episodes if e.get("generation") is not None]

    cards = "".join(
        [
            card("episodes", len(episodes)),
            card("emulators", instances),
            *( [card("generation", max(episode_generations))] if episode_generations else [] ),
            card("completed generations", len(generation_records)),
            card("verified laps", len(full_laps)),
            card(
                f"lap rate (last {len(lap_window)})",
                f"{len(recent_full_laps) * 100.0 / len(lap_window):.1f}%",
            ),
            card(
                "fastest verified lap",
                f"{min(e['episode_frames'] for e in official_laps) / 60.0:.2f}s"
                if official_laps else "none",
            ),
            card("best distance" if has_signed_distance else
                 "best gain" if uses_turn_slots else "best cp", max(progress_gain)),
            card(("gain" if uses_turn_slots else "cp") + " (last %d)" % len(recent),
                 f"{sum(e.get('net_progress', e.get('progress_gain', e.get('furthest', 0))) for e in recent) / len(recent):.1f}"),
            card(
                "start slots",
                ",".join(
                    str(slot)
                    for slot in sorted(
                        {e.get("start_slot") for e in episodes if e.get("start_slot") is not None}
                    )
                ) or "default",
            ),
            card("wall %", f"{sum(w for w in wall[-len(recent):]) / len(recent):.1f}"),
            card("centre %", f"{sum(centre[-len(recent):]) / len(recent):.1f}"),
            card("wall risk %", f"{sum(risk[-len(recent):]) / len(recent):.1f}"),
            card("assist %", f"{sum(assist[-len(recent):]) / len(recent):.1f}"),
            card("lean actual %", f"{sum(actual_lean[-len(recent):]) / len(recent):.1f}"),
            card("lean requested %", f"{sum(requested_lean[-len(recent):]) / len(recent):.1f}"),
            card("frames/action", f"{sum(frame_hold[-len(recent):]) / len(recent):.2f}"),
            card("turn quality %", f"{sum(turn_quality[-len(recent):]) / len(recent):.1f}"),
            card("policy agreement %", f"{sum(policy_agreement[-len(recent):]) / len(recent):.1f}"),
            card("ray clearance %", f"{sum(ray_clearance[-len(recent):]) / len(recent):.1f}"),
            card("ray response %", f"{sum(ray_response[-len(recent):]) / len(recent):.1f}"),
            card("toward wall %", f"{sum(toward_wall[-len(recent):]) / len(recent):.1f}"),
            card("speed loss/frame", f"{sum(speed_loss[-len(recent):]) / len(recent):.2f}"),
            card("reward", f"{sum(r for r in reward[-len(recent):]) / len(recent):.0f}"),
        ]
    )

    return (
        head
        + f'<div class="cards">{cards}</div>'
        + improvement_panel(episodes, assess_window)
        + lap_leaderboard(episodes)
        + lineage_panel(generation_records)
        + f"<h2>{progress_label}</h2>"
        + line_chart(progress_gain, COLOURS["furthest"], progress_label)
        + '<p class="note">Measured from each episode start. New time-trial '
        "runs use signed distance, so reversing reduces the score.</p>"
        + "<h2>wall contact</h2>"
        + line_chart(wall, COLOURS["wall"], "wall contact %", "{:.0f}%")
        + '<p class="note">The measured bottleneck: contact drains power, and '
        "power is what ends runs.</p>"
        + "<h2>track position</h2>"
        + line_chart(centre, COLOURS["centre"], "centre quality %", "{:.0f}%")
        + '<p class="note">100% is centred. This is a penalty-free view of the '
        "same geometry used by genetic tie-breaking.</p>"
        + "<h2>predicted wall risk</h2>"
        + line_chart(risk, COLOURS["risk"], "predicted wall risk %", "{:.0f}%")
        + '<p class="note">Rises before contact by combining current clearance '
        "with how quickly the wall is closing.</p>"
        + "<h2>safety intervention</h2>"
        + line_chart(assist, COLOURS["assist"], "safety override %", "{:.0f}%")
        + '<p class="note">This should fall as genomes learn to request safe '
        "actions themselves. A high value means the guard rails are carrying the policy.</p>"
        + "<h2>actual lean sent to the game</h2>"
        + line_chart(actual_lean, COLOURS["lean"], "actual lean %", "{:.0f}%")
        + "<h2>lean requested by the genome</h2>"
        + line_chart(requested_lean, COLOURS["requested"], "requested lean %", "{:.0f}%")
        + '<p class="note">The gap between these charts is the safety layer. '
        "Actual lean counts every emulated frame, including look-ahead corrections.</p>"
        + "<h2>turn outcome quality</h2>"
        + line_chart(turn_quality, COLOURS["turn"], "turn quality %", "{:.0f}%")
        + '<p class="note">A map-independent tie-breaker built from clearance, '
        "centering, falling wall risk, and safe movement near the episode frontier.</p>"
        + "<h2>genome agreement with safe controls</h2>"
        + line_chart(policy_agreement, COLOURS["agreement"], "policy agreement %", "{:.0f}%")
        + '<p class="note">Should rise as the genome learns corrections that the '
        "guard rail previously had to apply for it.</p>"
        + "<h2>seven-ray forward clearance</h2>"
        + line_chart(ray_clearance, COLOURS["rays"], "closest ray clearance %", "{:.0f}%")
        + '<p class="note">Mean closest forward ray per episode. Higher values '
        "mean the road fan stays open farther ahead of the car.</p>"
        + "<h2>ray response score</h2>"
        + line_chart(ray_response, COLOURS["ray_response"], "ray response %", "{:.0f}%")
        + '<p class="note">Positive means the genome steers away from the closer '
        "ray fan; negative means it steers toward the nearer wall.</p>"
        + "<h2>steering toward the closer wall</h2>"
        + line_chart(toward_wall, COLOURS["toward_wall"], "toward wall %", "{:.0f}%")
        + '<p class="note">Lower is better. These frames receive a direct penalty.</p>'
        + "<h2>ray clearance recovery</h2>"
        + line_chart(clearance_recovery, COLOURS["ray_response"], "clearance gain %", "{:.1f}%")
        + '<p class="note">How much forward clearance opens after a danger response.</p>'
        + "<h2>speed lost per frame</h2>"
        + line_chart(speed_loss, COLOURS["speed"], "raw speed loss/frame", "{:.1f}")
        + "<h2>frames per policy action</h2>"
        + line_chart(frame_hold, COLOURS["assist"], "frames per action", "{:.1f}")
        + '<p class="note">Normally four. It falls to two or one when a curve '
        "or approaching wall needs quicker decisions.</p>"
        + "<h2>reward per episode</h2>"
        + line_chart(reward, COLOURS["reward"], "reward", "{:.0f}", baseline_zero=True)
        + "<h2>mean speed</h2>"
        + line_chart(speed, COLOURS["speed"], "mean speed")
        + "<h2>where the reward came from</h2>"
        + reward_parts_table(episodes)
        + '<p class="note">Faint line is per episode, solid is a '
        f"{SMOOTH_WINDOW}-episode mean. Run-to-run variance here is large "
        "enough to fake a result -- identical configs have scored 81 and 66 -- "
        "so read the trend, and compare configurations with "
        "tools/compare_configs.py --repeats N.</p>"
        + "</body></html>"
    )


class Handler(BaseHTTPRequestHandler):
    pinned = None

    def do_GET(self):
        runs = list_runs()
        if not runs:
            body = (
                "<!doctype html><html><head><meta charset='utf-8'>"
                f"<meta http-equiv='refresh' content='{REFRESH_SECONDS}'>"
                f"<style>{STYLE}</style></head><body><h1>no runs recorded</h1>"
                "<p class='sub'>Start one:<br><br>"
                "&nbsp;&nbsp;python training/record_teacher.py --instances 8<br>"
                "&nbsp;&nbsp;train_rl.bat 8 500000</p></body></html>"
            )
        else:
            query = parse_qs(urlsplit(self.path).query)
            wanted = query.get("run", [None])[0]
            try:
                assess_window = int(query.get("window", [DEFAULT_ASSESS_WINDOW])[0])
            except (TypeError, ValueError):
                assess_window = DEFAULT_ASSESS_WINDOW
            assess_window = max(
                MIN_ASSESS_WINDOW, min(assess_window, MAX_ASSESS_WINDOW)
            )
            run_id = wanted or self.pinned or runs[0]
            if run_id not in runs:
                run_id = runs[0]
            body = render(run_id, runs, assess_window)

        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        # Silent: it reloads every few seconds and the log is pure noise.
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None, help="run id (default: newest)")
    parser.add_argument("--port", type=int, default=8321)
    args = parser.parse_args()

    Handler.pinned = args.run

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard on http://localhost:{args.port}  (Ctrl+C to stop)")
    runs = list_runs()
    print(f"{len(runs)} run(s) in the store" + (f", newest {runs[0]}" if runs else ""))

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
