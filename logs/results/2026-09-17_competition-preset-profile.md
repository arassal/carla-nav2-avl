# Competition preset: 2.6 Hz at 232% CPU, and where it goes

**Date:** 2026-09-17 · **Car:** dinosaur · Alexander's `148be12`
(white-line + pothole detection), measured against the same three cameras with
only the costmap node restarted.

| preset | costmap rate | costmap_node CPU |
|---|---|---|
| `perception_dinosaur.yaml` | 9.93 Hz | 67% |
| **`perception_competition.yaml`** | **2.59 Hz** | **232%** |

The competition preset cannot hold its 10 Hz `publish_rate`: it runs at a
quarter of it, so Nav2 would see an obstacle picture refreshed every ~0.4 s.

## Profile (cProfile on the node, 84 ticks)

    _tick                                451.6 ms/tick
      line_bev.detect_bev  (3 calls)     416.7   92%
        _ridge_contrast   (~66 calls)    330.2   73%

Inside `_ridge_contrast`, by self time per tick:

    numpy ufunc.reduce   (36389 calls)   155.9 ms   .sum() / .mean() over full frames
    cv2.warpAffine       (11184 calls)    95.9 ms   _shift, twice per call
    _ridge_contrast itself                64.2 ms

## Why it is expensive

`_ridge_contrast` is called once per candidate component (~22 per camera) and
each call works on the **whole fine grid**. With `bev_upsample: 4` over a
20 x 20 m costmap at 0.1 m, that grid is 800 x 800:

- `comp = _fill(shape, c)` allocates and fills 800 x 800 per component,
- `_shift` runs `cv2.warpAffine` over 800 x 800, twice per component,
- `comp.sum()`, `val[comp].mean()`, `side.sum()`, `val[side].mean()` all
  reduce over 800 x 800.

All of it to sample a thin strip either side of one small line segment.

## Suggested fix (not implemented -- for Alexander or a later session)

Crop to the component's bounding box, grown by the shift distance
`pad = ceil(max(2, 3 * line_px)) + 2`. `cv2.findContours` already gives the
box via `cv2.boundingRect`, and the ridge test never reads a pixel outside
it, so the answer is unchanged:

    x, y, w, h = cv2.boundingRect(c)
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
    comp_roi = fill(contour - [x0, y0]) into a (y1-y0, x1-x0) buffer
    _ridge_contrast(val[y0:y1, x0:x1], comp_roi, ..., valid[y0:y1, x0:x1])
    lines_fine[y0:y1, x0:x1] |= comp_roi

A component's window is typically tens of pixels across against 800, so this
should remove most of the 330 ms. Worth measuring `bev_upsample: 3` as well
(the fine grid scales with its square) and checking whether the two
`.mean()` calls can be replaced with one pass.

**Care:** this is the detector Alexander built for the competition course. Any
change needs an equivalence check against recorded frames, not just a speed
number.
