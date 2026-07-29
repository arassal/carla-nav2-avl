# Cone detection — how it works and how to make it better

For teammates. Everything here runs from `driving_seg/`; commands assume
that directory.

## The problem

The costmap needs cones (and painted white course lines) as *areas*, not
boxes. No public pretrained model has a cone class with MASKS (COCO: no
cones; Open Images: no cone masks; FSOCO: gated; Roboflow: API keys) — but
a good free pretrained cone DETECTOR does exist (see v3 below), so the
current design detects with it and derives masks, while our own fine-tuned
model covers white lines. Retraining ours takes ~30 minutes whenever it
isn't good enough.

## The pipeline (three models, one overlay)

```
frame ─┬─ scene   yolo11n-seg (COCO)      → person, vehicle, sign, light
       ├─ road    YOLOPv2                 → drivable area, lane lines
       └─ course  yolo11n-seg fine-tuned  → cone, white_line   ← ours
                └──────── fusion.py ────────→ one colored overlay + masks
```

Cones render orange, priority just below person — a cone is never painted
over by road/vehicle. `driving_seg/config.py` is the single source of truth
for classes/colors/priorities.

## How the training set is made (self-distillation)

No hand labeling. `tools/build_course_dataset.py`:

1. **Scrape** freely-licensed photos (Wikimedia Commons API; competition
   terms included: autocross, gymkhana, slalom, Formula Student).
2. **Auto-label** with classical CV: orange HSV gate (plus a pale band for
   sun-faded cones) + contour geometry (taller-than-wide, solid). Painted
   lines: white gate restricted to grass surroundings, elongated shapes.
3. **Multiply** with `tools/augment_cones.py`: every labeled cone is cut out
   and composited onto varied backgrounds at random scale/flip/brightness —
   pixel-perfect labels by construction (copy-paste augmentation).
4. **Fine-tune** with `tools/train_course.py` (yolo11n-seg base;
   `overlap_mask=False mask_ratio=1` keeps thin line masks crisp).

The network generalizes past the color heuristic that labeled it — that's
the self-distillation bet, and it's why v2 catches faded cones the HSV gate
alone would miss.

## Current status (v1 → v2)

- v1 (93 scraped images): mask mAP50 ≈ 0.29 — solid on classic orange
  cones, misses striped/faded ones.
- v2 (103 real + 299 copy-paste synthetic, faded-orange band): cone mask
  AP50 0.295, white_line 0.431. Orange cones solid; striped cones weak —
  the color-gate teacher can't label what it can't see.
- **v3 (current): hybrid.** A dedicated pretrained cone DETECTOR
  (ExStella/Traffic-cones, Hugging Face, Apache-2.0 — models/cone_det.pt,
  committed) finds every cone incl. striped/white-banded ones; each box is
  converted to an area mask (color gate + convex hull inside the box, box
  fallback). Our fine-tuned model keeps white_line. Verified on the
  striped-cone test image: full cluster highlighted. Pipeline still 65 FPS
  on the 5090.
- `white_line` is undertrained (public grass-line imagery is scarce) — the
  fix is the section below.

## The single highest-value thing you can do

**Photograph OUR cones and OUR course lines with the car's cameras** —
50–100 shots, varied distance/angle/light — then:

    # drop photos into datasets/course/images/train + label, or let the
    # auto-labeler do it: put them in a folder and adapt CONE_TERMS scraping
    # aside, the simplest path is auto-labeling in place:
    python3 - <<'EOF'
    # auto-label a folder of your own photos into the dataset
    import glob, cv2, sys; sys.path.insert(0, "tools")
    from build_course_dataset import label_cones, label_white_lines, write_sample
    for i, p in enumerate(glob.glob("my_photos/*.jpg")):
        img = cv2.imread(p)
        labels = [(0, poly) for poly in label_cones(img)] + \
                 [(1, poly) for poly in label_white_lines(img)]
        if labels:
            write_sample(img, labels, "train", "ours_%03d" % i)
    EOF
    python3 tools/augment_cones.py --count 300
    python3 tools/train_course.py --epochs 100      # ~20 min on the 5090

Spot-check `datasets/course/review_cones.jpg` (label contact sheet) before
training — bad labels in, bad model out. Real photos of the actual
competition hardware beat any amount of internet data.

## Deploying to the car

    python3 tools/export_trt.py          # on the Jetson: builds .engine
    # then point --course-weights (or the wrapper default) at models/course.engine

## Gotchas we already hit (so you don't)

- Wikimedia throttles bulk downloads unless the User-Agent has contact info
  (the builder's UA is compliant — keep it that way).
- `getPerspectiveTransform`-style silent failures aside, the recurring
  lesson: **verify with asymmetric fixtures** — a model/dataset change gets
  eyeballed on `datasets/course/review_cones.jpg` and one hard test image
  (`testdata/cones/`), not trusted from metrics alone.
- Thin masks (lines) die at default mask downsampling — keep the
  `overlap_mask=False, mask_ratio=1` training flags.
