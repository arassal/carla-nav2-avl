"""Copy-paste augmentation for the cone class.

Cuts every labeled cone out of datasets/course train images and composites
2-6 of them per synthetic frame onto varied backgrounds (testdata/street,
testdata/grass_lines) at random scale/position/flip/brightness. Labels are
exact by construction. Appends `synth_*` samples to the train split.

    python3 tools/augment_cones.py --count 300
"""

import argparse
import glob
import os
import random

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "datasets", "course")
BG_DIRS = ["testdata/street", "testdata/grass_lines", "testdata/cones"]


def load_cone_crops():
    crops = []
    for lab in glob.glob(os.path.join(ROOT, "labels", "train", "cone_*.txt")):
        img = cv2.imread(lab.replace("labels", "images").replace(".txt", ".jpg"))
        if img is None:
            continue
        ih, iw = img.shape[:2]
        for line in open(lab):
            parts = line.split()
            if not parts or parts[0] != "0":
                continue
            poly = np.array(parts[1:], float).reshape(-1, 2) * [iw, ih]
            poly = poly.astype(np.int32)
            x, y, w, h = cv2.boundingRect(poly)
            if w < 12 or h < 16:
                continue
            mask = np.zeros((ih, iw), np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            crops.append((img[y:y + h, x:x + w].copy(),
                          mask[y:y + h, x:x + w].copy()))
    return crops


def synth(bg, crops, k):
    h, w = bg.shape[:2]
    out = bg.copy()
    labels = []
    for _ in range(k):
        crop, cmask = random.choice(crops)
        s = random.uniform(0.35, 1.6) * (w / 1280.0)
        ch, cw = max(16, int(crop.shape[0] * s)), max(12, int(crop.shape[1] * s))
        if ch >= h // 2 or cw >= w // 2:
            continue
        c = cv2.resize(crop, (cw, ch))
        m = cv2.resize(cmask, (cw, ch), interpolation=cv2.INTER_NEAREST)
        if random.random() < 0.5:
            c, m = c[:, ::-1], m[:, ::-1]
        c = np.clip(c.astype(np.int16) + random.randint(-40, 25), 0, 255).astype(np.uint8)
        # ground-plausible placement: lower 60% of frame
        y0 = random.randint(int(h * 0.4), h - ch - 1)
        x0 = random.randint(0, w - cw - 1)
        roi = out[y0:y0 + ch, x0:x0 + cw]
        sel = m > 127
        roi[sel] = c[sel]
        cnts, _ = cv2.findContours((m > 127).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cn in cnts:
            if cv2.contourArea(cn) < 80:
                continue
            p = (cn.reshape(-1, 2) + [x0, y0]).astype(float) / [w, h]
            labels.append("0 " + " ".join("%.5f" % v for v in p.reshape(-1)))
    return out, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=300)
    args = ap.parse_args()
    random.seed(11)

    crops = load_cone_crops()
    bgs = [p for d in BG_DIRS for p in glob.glob(d + "/*.jpg")]
    print("cone crops:", len(crops), "| backgrounds:", len(bgs))
    if not crops or not bgs:
        raise SystemExit("need labeled cones + backgrounds first")

    made = 0
    for i in range(args.count):
        bg = cv2.imread(random.choice(bgs))
        if bg is None:
            continue
        out, labels = synth(bg, crops, random.randint(2, 6))
        if not labels:
            continue
        stem = "synth_%04d" % i
        cv2.imwrite(os.path.join(ROOT, "images", "train", stem + ".jpg"), out)
        with open(os.path.join(ROOT, "labels", "train", stem + ".txt"), "w") as f:
            f.write("\n".join(labels) + "\n")
        made += 1
    print("synthetic samples written:", made)


if __name__ == "__main__":
    main()
