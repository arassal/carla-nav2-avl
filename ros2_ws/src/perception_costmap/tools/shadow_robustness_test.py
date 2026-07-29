#!/usr/bin/env python3
"""Shadow robustness test for the road segmenter.

Measures how much detected road survives inside a synthetic shadow, using the
same segmenter the costmap node runs. Re-run after any model swap or retrain
(results 2026-07-13, twinlite_nano.engine, 5 mixed live+recorded frames):

    brightness x0.60  kept 99.9%   (typical cloud/building shadow)
    brightness x0.45  kept 99.8%   (strong afternoon shadow)
    brightness x0.30  kept 93.9%   (deep shadow; control itself 97.5%)
    brightness x0.18  kept 82.1%   (near-dusk dark; control 89.4%)

Interpretation: realistic shadows are a non-issue; losses at extreme darkness
are scattered fringe pixels (absorbed by the temporal filter + BEV threshold),
not region flips. Known weak spot: deep shadow COMBINED with long distance
under-segments -- benign inside the 12 m trusted range / FOV trim, but watch
dusk/dawn. Wet pavement and sun glare are NOT covered by this test.

Usage:
    # collect a few representative frames first, e.g.
    #   curl 'http://<jetson>:8080/snapshot?topic=/zed_front/...' -o frames/f1.jpg
    python3 tools/shadow_robustness_test.py frames/*.jpg [--out /tmp/shadow_out]

Writes per-factor overlays (lost road = red) and per-frame real-shadow
overlays (road = green tint) into --out for eyeballing alongside the numbers.
"""
import argparse
import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from perception_costmap import segmentation

FACTORS = (0.6, 0.45, 0.3, 0.18)


def synth_shadow(img, factor, soft_px=25):
    """Soft-edged polygonal shadow band across the road region, darkened by
    *factor*, blue channel dimmed least (sky-lit shadows are blue-shifted)."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    pts = np.array([[int(w*0.05), h], [int(w*0.45), int(h*0.35)],
                    [int(w*0.75), int(h*0.35)], [int(w*0.55), h]], np.int32)
    cv2.fillPoly(mask, [pts], 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), soft_px)
    shade = np.ones(3, np.float32) * factor
    shade[0] = min(1.0, factor * 1.15)          # BGR: blue dimmed least
    out = img.astype(np.float32)
    out = out * (1 - mask[..., None]) + out * shade * mask[..., None]
    return out.astype(np.uint8), mask > 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('frames', nargs='+', help='input image files')
    ap.add_argument('--out', default='/tmp/shadow_out', help='overlay dir')
    ap.add_argument('--weights',
                    default='/home/dinosaur/models/twinlite_nano.engine')
    ap.add_argument('--repo',
                    default='/home/dinosaur/models/TwinLiteNetPlus')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    seg = segmentation.create_segmenter(
        'twinlitenet', repo_path=args.repo, weights=args.weights,
        config='nano')

    results = {}
    for idx, f in enumerate(args.frames):
        img = cv2.imread(f)
        if img is None:
            print(f'skip unreadable {f}')
            continue
        base = seg(img).astype(bool)

        # real-shadow overlay for eyeballing (green = road)
        vis = img.copy()
        vis[base] = vis[base] // 2 + np.array((0, 128, 0), np.uint8)
        vis[~base] = vis[~base] // 2 + np.array((0, 0, 128), np.uint8)
        cv2.imwrite(os.path.join(args.out, f'real_{idx}.jpg'), vis)

        for factor in FACTORS:
            shadowed, patch = synth_shadow(img, factor)
            m = seg(shadowed).astype(bool)
            roi = patch & base
            if roi.sum() < 500:
                continue
            kept = (m & roi).sum() / roi.sum()
            ctrl = base & ~patch
            ctrl_kept = (m & ctrl).sum() / max(int(ctrl.sum()), 1)
            results.setdefault(factor, []).append((kept, ctrl_kept))
            if idx == 0:
                ov = shadowed.copy()
                ov[base & ~m] = (0, 0, 255)          # lost road -> red
                ov[m & base] = ov[m & base] // 2 + np.array((0, 128, 0), np.uint8)
                cv2.imwrite(os.path.join(
                    args.out, f'overlay_{int(factor*100)}.jpg'), ov)

    print()
    print('road retention INSIDE synthetic shadow (1.0 = unaffected):')
    for factor in sorted(results, reverse=True):
        rs = results[factor]
        k = float(np.mean([a for a, _ in rs]))
        c = float(np.mean([b for _, b in rs]))
        print(f'  brightness x{factor:<5} kept {k*100:5.1f}%'
              f'  (control outside patch: {c*100:5.1f}%)  n={len(rs)}')
    print(f'overlays -> {args.out}')


if __name__ == '__main__':
    main()
