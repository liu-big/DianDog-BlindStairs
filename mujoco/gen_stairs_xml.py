#!/usr/bin/env python3
"""Generate diandog_stairs.xml for MuJoCo play.

Default: large bright ground + two parallel strips (15cm and 20cm).
Optional play-only tweaks (--chamfer / --foot-r / --no-hfe) for MuJoCo contact.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FLAT = ROOT / "diandog_flat.xml"


def build_strip_geoms(
    *,
    prefix: str,
    step_h: float,
    y_center: float,
    tread: float = 0.30,
    start_len: float = 6.0,
    steps: int = 8,
    pattern: tuple[str, ...] = ("up", "down"),
    end_len: float = 4.0,
    width: float = 2.4,
    ground_thickness: float = 0.10,
    chamfer: float = 0.0,
    rgba: str = "0.55 0.58 0.62 1",
) -> tuple[str, float, float]:
    """One stairs strip along +X, centered at y_center. Returns (xml, spawn_x, total_len)."""
    step_len = float(tread)
    seg_len = step_len * steps
    lines: list[str] = []
    total_len = start_len + len(pattern) * seg_len + end_len
    fric = 'friction="1.5 0.005 0.0001"'
    yc = float(y_center)
    half_w = 0.5 * width

    def solid_span(x0: float, length: float, top: float, name: str) -> None:
        top = max(top, 0.0)
        box_h = top + ground_thickness
        zc = 0.5 * top - 0.5 * ground_thickness
        lines.append(
            f'    <geom name="{prefix}{name}" type="box" '
            f'pos="{x0 + 0.5 * length:.4f} {yc:.4f} {zc:.4f}" '
            f'size="{0.5 * length:.4f} {half_w:.4f} {0.5 * box_h:.4f}" '
            f'rgba="{rgba}" {fric}/>'
        )

    def riser_chamfer(x_face: float, z_lo: float, z_hi: float, name: str) -> None:
        dh = max(float(z_hi - z_lo), 0.0)
        if chamfer <= 1e-6 or dh < 1e-4:
            return
        run = min(float(chamfer), 0.45 * step_len, dh)
        half = 0.5 * run * 1.41421356237
        cx = x_face + 0.5 * run
        cz = z_lo + 0.5 * run
        lines.append(
            f'    <geom name="{prefix}{name}" type="box" '
            f'pos="{cx:.4f} {yc:.4f} {cz:.4f}" '
            f'size="{half:.4f} {half_w:.4f} {half:.4f}" '
            f'quat="0.9238795 0 0.3826834 0" rgba="0.62 0.64 0.68 1" {fric}/>'
        )

    x = 0.0
    h = 0.0
    solid_span(x, start_len, h, "start")
    x += start_len
    si = 0
    for seg in pattern:
        for _ in range(steps):
            prev_h = h
            h = h + step_h if seg == "up" else h - step_h
            solid_span(x, step_len, max(h, 0.0), f"step_{si}")
            if seg == "up" and h > prev_h:
                riser_chamfer(x, prev_h, h, f"chamfer_{si}")
            x += step_len
            si += 1
    while h > 0.5 * step_h:
        h = max(h - step_h, 0.0)
        solid_span(x, step_len, h, f"step_{si}")
        x += step_len
        si += 1
    solid_span(x, end_len, 0.0, "end")
    return "\n".join(lines), 0.5 * start_len, total_len


def build_stairs_geoms(
    *,
    step_h: float,
    tread: float = 0.30,
    start_len: float = 5.0,
    steps: int = 8,
    pattern: tuple[str, ...] = ("up", "down", "up", "down"),
    end_len: float = 4.0,
    width: float = 3.0,
    ground_thickness: float = 0.10,
    chamfer: float = 0.0,
) -> tuple[str, float]:
    """Legacy single-strip API."""
    xml, spawn_x, _ = build_strip_geoms(
        prefix="",
        step_h=step_h,
        y_center=0.0,
        tread=tread,
        start_len=start_len,
        steps=steps,
        pattern=pattern,
        end_len=end_len,
        width=width,
        ground_thickness=ground_thickness,
        chamfer=chamfer,
    )
    return xml, spawn_x


def build_dual_stairs(
    *,
    tread: float = 0.30,
    steps: int = 8,
    start_len: float = 8.0,
    end_len: float = 6.0,
    strip_width: float = 2.6,
    lane_gap: float = 1.2,
    ground_thickness: float = 0.12,
    chamfer: float = 0.05,
    ground_pad: float = 8.0,
) -> tuple[str, float, float]:
    """15cm (y+) and 20cm (y-) strips on a large bright ground. Returns xml, spawn_x, spawn_y."""
    half_gap = 0.5 * lane_gap + 0.5 * strip_width
    y15 = +half_gap
    y20 = -half_gap
    fric = 'friction="1.5 0.005 0.0001"'

    xml15, spawn_x, len15 = build_strip_geoms(
        prefix="s15_",
        step_h=0.15,
        y_center=y15,
        tread=tread,
        start_len=start_len,
        steps=steps,
        pattern=("up", "down"),
        end_len=end_len,
        width=strip_width,
        ground_thickness=ground_thickness,
        chamfer=chamfer,
        rgba="0.52 0.62 0.72 1",
    )
    xml20, _, len20 = build_strip_geoms(
        prefix="s20_",
        step_h=0.20,
        y_center=y20,
        tread=tread,
        start_len=start_len,
        steps=steps,
        pattern=("up", "down"),
        end_len=end_len,
        width=strip_width,
        ground_thickness=ground_thickness,
        chamfer=chamfer,
        rgba="0.62 0.55 0.48 1",
    )
    total_len = max(len15, len20)
    total_w = 2.0 * half_gap + strip_width + 2.0 * ground_pad
    # Large bright checker floor under everything
    floor = (
        f'    <geom name="ground_big" type="box" '
        f'pos="{0.5 * total_len:.4f} 0 {-0.5 * ground_thickness:.4f}" '
        f'size="{0.5 * total_len + ground_pad:.4f} {0.5 * total_w:.4f} {0.5 * ground_thickness:.4f}" '
        f'material="groundplane" {fric}/>\n'
        # Mid flat lane between strips (spawn corridor)
        f'    <geom name="mid_lane" type="box" '
        f'pos="{0.5 * start_len:.4f} 0 {-0.5 * ground_thickness + 0.001:.4f}" '
        f'size="{0.5 * start_len:.4f} {0.5 * lane_gap:.4f} {0.5 * ground_thickness:.4f}" '
        f'rgba="0.70 0.72 0.75 1" {fric}/>'
    )
    # Bright lights
    lights = (
        '    <light name="key" pos="6 0 8" dir="0 0 -1" diffuse="1.0 1.0 1.0" specular="0.3 0.3 0.3" cutoff="80"/>\n'
        '    <light name="fill" pos="-2 4 6" dir="0.2 -0.3 -1" diffuse="0.7 0.75 0.8" cutoff="90"/>\n'
        '    <light name="rim" pos="10 -4 5" dir="-0.3 0.2 -1" diffuse="0.55 0.55 0.6" cutoff="90"/>'
    )
    # Visual labels (non-colliding)
    labels = (
        f'    <geom name="label_15" type="box" pos="1.0 {y15:.4f} 0.35" '
        f'size="0.35 0.08 0.02" contype="0" conaffinity="0" rgba="0.2 0.55 0.95 1"/>\n'
        f'    <geom name="label_20" type="box" pos="1.0 {y20:.4f} 0.35" '
        f'size="0.35 0.08 0.02" contype="0" conaffinity="0" rgba="0.95 0.45 0.2 1"/>'
    )
    xml = "\n".join([lights, floor, xml15, xml20, labels])
    # Spawn on mid lane facing +X, between the two strips
    return xml, spawn_x, 0.0


def _disable_leg_boxes(xml: str, *, bodies: str = "hfe") -> str:
    body_re = re.compile(rf'<body name="(?:fl|fr|rl|rr)_({bodies})"')
    any_body = re.compile(r"<body\b")
    stack: list[bool] = []
    out: list[str] = []
    for ln in xml.splitlines():
        if any_body.search(ln):
            stack.append(bool(body_re.search(ln)))
        if (
            stack
            and stack[-1]
            and 'type="box"' in ln
            and "contype=" not in ln
            and 'rgba="0 0 0 0"/>' in ln
        ):
            ln = ln.replace(
                'rgba="0 0 0 0"/>',
                'contype="0" conaffinity="0" rgba="0 0 0 0"/>',
            )
        out.append(ln)
        if stack and "</body>" in ln:
            stack.pop()
    return "\n".join(out)


def _brighten_assets(xml: str) -> str:
    """Lighter checker + brighter default material."""
    xml = xml.replace(
        'rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"',
        'rgb1="0.55 0.58 0.62" rgb2="0.42 0.45 0.50"',
    )
    xml = xml.replace('texrepeat="5 5"', 'texrepeat="8 8"')
    return xml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step-h", type=float, default=0.20, help="single-strip height (ignored if --dual)")
    ap.add_argument("--dual", action="store_true", help="15cm + 20cm parallel strips (default play)")
    ap.add_argument("--tread", type=float, default=0.30)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--start-len", type=float, default=8.0)
    ap.add_argument("--end-len", type=float, default=6.0)
    ap.add_argument("--chamfer", type=float, default=0.05, help="riser chamfer run (m), 0=off")
    ap.add_argument("--foot-r", type=float, default=0.020, help="foot sphere radius")
    ap.add_argument("--no-hfe", action="store_true", help="disable thigh collision boxes")
    ap.add_argument("--no-kfe", action="store_true", help="disable shank collision boxes")
    ap.add_argument("-o", type=Path, default=ROOT / "diandog_stairs.xml")
    args = ap.parse_args()
    # Dual is the default play layout unless user sets --step-h alone without --dual;
    # play_stairs.sh always passes --dual.
    use_dual = bool(args.dual)

    flat = FLAT.read_text()
    if '<geom name="floor"' not in flat:
        raise SystemExit("unexpected flat xml: no floor geom")
    flat_nofloor = "\n".join(ln for ln in flat.splitlines() if 'name="floor"' not in ln)
    flat_nofloor = _brighten_assets(flat_nofloor)
    if abs(args.foot_r - 0.015) > 1e-9:
        flat_nofloor = flat_nofloor.replace(
            'size="0.015" rgba="0 0 0 0"/>',
            f'size="{args.foot_r:.3f}" rgba="0 0 0 0"/>',
        )
    disable = []
    if args.no_hfe:
        disable.append("hfe")
    if args.no_kfe:
        disable.append("kfe")
    if disable:
        flat_nofloor = _disable_leg_boxes(flat_nofloor, bodies="|".join(disable))

    # Remove the single dim light; dual builder injects bright lights
    flat_nofloor = "\n".join(
        ln for ln in flat_nofloor.splitlines() if not (ln.strip().startswith("<light") and "pos=" in ln)
    )

    if use_dual:
        stairs, spawn_x, spawn_y = build_dual_stairs(
            tread=args.tread,
            steps=args.steps,
            start_len=args.start_len,
            end_len=args.end_len,
            chamfer=args.chamfer,
        )
        desc = "dual 15cm(+y) + 20cm(-y)"
    else:
        stairs, spawn_x = build_stairs_geoms(
            step_h=args.step_h,
            tread=args.tread,
            steps=args.steps,
            start_len=args.start_len,
            end_len=args.end_len,
            chamfer=args.chamfer,
            width=4.0,
        )
        spawn_y = 0.0
        desc = f"single step_h={args.step_h:.3f}"

    # Insert after <worldbody>
    if "<worldbody>" not in flat_nofloor:
        raise SystemExit("worldbody missing")
    body = flat_nofloor.replace("<worldbody>", "<worldbody>\n" + stairs, 1)

    spawn_z = 0.48
    body = body.replace(
        '<body name="base_link" pos="0 0 0.48">',
        f'<body name="base_link" pos="{spawn_x:.4f} {spawn_y:.4f} {spawn_z:.2f}">',
        1,
    )
    body = body.replace(
        'qpos="0 0 0.48 1 0 0 0',
        f'qpos="{spawn_x:.4f} {spawn_y:.4f} {spawn_z:.2f} 1 0 0 0',
        1,
    )
    while body.lstrip().startswith("<!--"):
        body = body.split("\n", 1)[1]
    header = (
        f"<!-- DianDog stairs: {desc} tread={args.tread:.3f} steps/seg={args.steps} "
        f"start={args.start_len:.1f} end={args.end_len:.1f} "
        f"spawn=({spawn_x:.3f},{spawn_y:.3f}) chamfer={args.chamfer:.3f} "
        f"foot_r={args.foot_r:.3f} no_hfe={int(args.no_hfe)} -->\n"
    )
    args.o.write_text(header + body.lstrip() + "\n")
    print(
        f"[gen] wrote {args.o}  {desc}  tread={args.tread:.3f}m  "
        f"steps/seg={args.steps}  start={args.start_len:.1f}m  end={args.end_len:.1f}m  "
        f"spawn=({spawn_x:.3f},{spawn_y:.3f}) chamfer={args.chamfer:.3f} foot_r={args.foot_r:.3f}"
    )


if __name__ == "__main__":
    main()
