"""Build the course-model training set by self-distillation.

Scrapes freely-licensed images (Wikimedia Commons API), auto-labels them
with classical CV, and writes a YOLO-seg dataset:

  cone:       orange HSV gate + contour geometry (upright, solid, small)
  white_line: whiteness gate restricted to green-ish (grass) surroundings,
              elongated components only

Auto-labels are noisy teachers; the network generalizes past them. Swap in
a curated dataset (FSOCO seg / Roboflow) later without touching training.

  python3 tools/build_course_dataset.py --per-term 40
  -> datasets/course/{images,labels}/{train,val}/ + review contact sheets
"""

import argparse
import json
import os
import random
import re
import time
import sys
import urllib.parse
import urllib.request

import cv2
import numpy as np

UA = {"User-Agent": "drivingseg/0.1 (https://github.com/arassal/carla-nav2-avl; contact: alexander@assalfamily.com) python-urllib"}
ROOT = os.path.join(os.path.dirname(__file__), "..", "datasets", "course")

CONE_TERMS = ["traffic cone", "traffic cones road", "orange safety cone",
              "traffic cones construction", "pylon traffic",
              "slalom cones driving", "traffic cone street",
              "autocross cones", "gymkhana cones", "motorkhana",
              "driving test cones car", "cone slalom competition",
              "Formula Student cones track", "robot competition cones",
              "parking course cones", "koenen slalom"]
LINE_TERMS = ["football pitch line grass", "soccer field white line",
              "sports field markings grass", "touchline grass",
              "baseball foul line grass", "rugby pitch lines"]


def search(term, limit):
    q = urllib.parse.urlencode(dict(action="query", list="search",
        srsearch=term, srnamespace=6, srlimit=limit, format="json"))
    req = urllib.request.Request(
        "https://commons.wikimedia.org/w/api.php?" + q, headers=UA)
    try:
        data = json.load(urllib.request.urlopen(req, timeout=30))
        return [r["title"] for r in data["query"]["search"]
                if r["title"].lower().endswith((".jpg", ".jpeg", ".png"))]
    except Exception:
        return []


def resolve_urls(titles, width=960):
    """Batch imageinfo query -> {title: direct thumb URL on the CDN}."""
    out = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        q = urllib.parse.urlencode(dict(
            action="query", prop="imageinfo", iiprop="url",
            iiurlwidth=width, titles="|".join(chunk), format="json"))
        try:
            req = urllib.request.Request(
                "https://commons.wikimedia.org/w/api.php?" + q, headers=UA)
            data = json.load(urllib.request.urlopen(req, timeout=30))
            for page in data["query"]["pages"].values():
                ii = page.get("imageinfo")
                if ii:
                    out[page["title"]] = ii[0].get("thumburl") or ii[0]["url"]
        except Exception:
            pass
        time.sleep(0.3)
    return out


def fetch(url, dest):
    time.sleep(1.5)
    try:
        blob = urllib.request.urlopen(
            urllib.request.Request(url, headers=UA), timeout=45).read()
        with open(dest, "wb") as f:
            f.write(blob)
        img = cv2.imread(dest)
        if img is None or min(img.shape[:2]) < 200:
            os.remove(dest)
            return None
        return img
    except Exception:
        if os.path.exists(dest):
            os.remove(dest)
        return None


# ---------------- auto-labelers: return list of polygons per class ----------

def label_cones(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # orange gate + pale band for sun-faded cones
    m = cv2.inRange(hsv, (2, 110, 80), (18, 255, 255))
    # faded / sun-bleached cones: low-sat pale orange band
    m |= cv2.inRange(hsv, (4, 55, 140), (22, 130, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    h, w = img.shape[:2]
    polys = []
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.0004 * h * w or area > 0.25 * h * w:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bh < bw * 0.7:                      # cones are taller than wide-ish
            continue
        hull = cv2.convexHull(c)
        if area / max(cv2.contourArea(hull), 1) < 0.55:   # solid blob
            continue
        polys.append(c.reshape(-1, 2))
    return polys


def label_white_lines(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    grass = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    if grass.mean() < 0.15 * 255:              # not a grass scene: no labels
        return []
    white = cv2.inRange(hsv, (0, 0, 165), (180, 70, 255))
    near_grass = cv2.dilate(grass, np.ones((25, 25), np.uint8))
    white &= near_grass
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    polys = []
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 0.0006 * h * w:
            continue
        rect = cv2.minAreaRect(c)
        (rw, rh) = rect[1]
        if min(rw, rh) == 0 or max(rw, rh) / max(min(rw, rh), 1) < 3.0:
            continue                            # lines are elongated
        polys.append(c.reshape(-1, 2))
    return polys


# ---------------- YOLO-seg writer ----------------

def write_sample(img, labels, split, stem):
    ih, iw = img.shape[:2]
    img_dir = os.path.join(ROOT, "images", split)
    lab_dir = os.path.join(ROOT, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), img)
    with open(os.path.join(lab_dir, stem + ".txt"), "w") as f:
        for cid, poly in labels:
            poly = poly.astype(np.float64)
            poly[:, 0] /= iw
            poly[:, 1] /= ih
            if len(poly) < 3:
                continue
            f.write(str(cid) + " " +
                    " ".join("%.5f" % v for v in poly.reshape(-1)) + "\n")


def contact_sheet(entries, path, n=48):
    random.shuffle(entries)
    tiles = []
    for img, labels in entries[:n]:
        t = cv2.resize(img, (240, 180))
        sx, sy = 240.0 / img.shape[1], 180.0 / img.shape[0]
        for cid, poly in labels:
            p = (poly * np.array([sx, sy])).astype(np.int32)
            cv2.polylines(t, [p], True,
                          (0, 140, 255) if cid == 0 else (255, 255, 255), 2)
        tiles.append(t)
    if not tiles:
        return
    rows = [np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles) - 5, 6)]
    if rows:
        cv2.imwrite(path, np.vstack(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-term", type=int, default=40)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()
    random.seed(7)

    jobs = [(CONE_TERMS, label_cones, 0, "cone"),
            (LINE_TERMS, label_white_lines, 1, "line")]
    review = {"cone": [], "line": []}
    counts = {"cone": 0, "line": 0, "skipped": 0, "fetch_fail": 0}
    seen = set()

    for terms, labeler, cid, tag in jobs:
        titles = []
        for term in terms:
            titles += [t for t in search(term, args.per_term)
                       if t not in seen and not seen.add(t)]
        urls = resolve_urls(titles)
        print(tag, "resolved", len(urls), "of", len(titles))
        for title in titles:
            if True:   # keeps diff-history indentation; no-op
                if title not in urls:
                    counts["fetch_fail"] += 1
                    continue
                stem = tag + "_" + re.sub(r"[^A-Za-z0-9]", "_",
                                          title.replace("File:", ""))[:48]
                tmp = "/tmp/ds_fetch_%d.jpg" % os.getpid()
                img = fetch(urls[title], tmp)
                if img is None:
                    counts["fetch_fail"] += 1
                    continue
                polys = labeler(img)
                if not polys:                   # unlabeled: skip (no negatives
                    counts["skipped"] += 1      # needed; COCO backgrounds vary)
                    continue
                labels = [(cid, p) for p in polys]
                split = "val" if random.random() < args.val_frac else "train"
                write_sample(img, labels, split, stem)
                review[tag].append((img, labels))
                counts[tag] += 1
        print(tag, "done:", counts[tag], "images")

    os.makedirs(ROOT, exist_ok=True)
    contact_sheet(review["cone"], os.path.join(ROOT, "review_cones.jpg"))
    contact_sheet(review["line"], os.path.join(ROOT, "review_lines.jpg"))
    with open(os.path.join(ROOT, "course.yaml"), "w") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\n"
                "names:\n  0: cone\n  1: white_line\n" % os.path.abspath(ROOT))
    print(counts)


if __name__ == "__main__":
    main()
