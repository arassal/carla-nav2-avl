"""
costmap.py
----------
Beginner-friendly costmap pipeline for a top-down (BEV) road image.

Pipeline:
  1. ROAD SEGMENTATION  -> drawn GREEN
     Classical approach: the road is usually a fairly uniform gray/dark
     region. We use a simple color-range threshold + "keep the largest
     connected blob" trick. This is crude but transparent -- you can see
     exactly why a pixel was classified as road.

  2. OBSTACLE DETECTION  -> drawn RED
     Two interchangeable detectors are provided:
       a) classical_obstacle_mask()  - contrast-based blob detector.
          Works with NO extra installs. Good enough for a synthetic
          test image or simple real footage.
       b) yolo_obstacle_mask()       - uses Ultralytics YOLO to find
          actual "car" boxes. Use this on your real machine where
          ultralytics is installed.

  3. COST GRID
     We downsample the image into an NxM grid of cells. Each cell gets
     a cost:
         0   = free road            (cheap to drive through)
         100 = obstacle             (forbidden)
         50  = off-road / unknown   (expensive, avoid if possible)

  4. PATH PLANNING -> drawn BLUE
     A* search across the cost grid, constrained to start on the LEFT
     edge and end on the RIGHT edge, picking the lowest-total-cost path.

Run:
    python3 costmap.py --image your_image.png --out result.png
    python3 costmap.py --demo   # generates+uses a synthetic test image
"""

import argparse
import heapq
import numpy as np
import cv2


# ----------------------------------------------------------------------
# 1. ROAD SEGMENTATION (green)
# ----------------------------------------------------------------------
def segment_road(img_bgr):
    """
    Returns a binary mask (uint8, 0/255) where 255 = "this pixel is road".

    Approach: roads are usually low-saturation gray asphalt. We threshold
    in HSV on saturation+value, then keep only the largest connected
    component (assumption: the road is the single biggest blob in the
    image once you remove cars/sidewalks/etc).

    Tune the HSV ranges below to match YOUR camera/lighting.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Asphalt: low saturation (not very colorful), mid-range brightness.
    sat_mask = s < 60
    val_mask = (v > 40) & (v < 200)
    road_mask = (sat_mask & val_mask).astype(np.uint8) * 255

    # Clean up noise
    kernel = np.ones((9, 9), np.uint8)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel)

    # Keep only the largest connected blob -> assume that's "the road"
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(road_mask)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        road_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    return road_mask


# ----------------------------------------------------------------------
# 2a. OBSTACLE DETECTION -- classical (no extra installs needed)
# ----------------------------------------------------------------------
def classical_obstacle_mask(img_bgr, road_mask, min_obstacle_area=150):
    """
    Finds blobs that sit ON TOP of the road but don't match the road's
    color (i.e. cars). Returns (obstacle_mask, road_extent) -- both
    binary masks, 255 = positive.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # "Not road-colored": either too saturated (colorful car paint)
    # or too dark/bright compared to asphalt.
    not_road_color = (s > 60) | (v < 40) | (v > 200)

    # Cars sitting on the road create "holes" in road_mask (their pixels
    # don't look like asphalt), so we can't just AND with road_mask --
    # that would exclude the very pixels we're looking for. Instead we
    # take the convex hull of the road region to get its full extent,
    # regardless of how large any car-shaped holes are.
    contours, _ = cv2.findContours(road_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    road_extent = np.zeros_like(road_mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(largest)
        cv2.drawContours(road_extent, [hull], -1, 255, thickness=cv2.FILLED)
    on_road_area = road_extent > 0

    obstacle_raw = (not_road_color & on_road_area).astype(np.uint8) * 255

    kernel = np.ones((5, 5), np.uint8)
    obstacle_raw = cv2.morphologyEx(obstacle_raw, cv2.MORPH_OPEN, kernel)
    obstacle_raw = cv2.dilate(obstacle_raw, kernel, iterations=1)

    # Drop tiny specks (noise), keep blobs that look like a car's footprint
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(obstacle_raw)
    clean = np.zeros_like(obstacle_raw)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_obstacle_area:  # tune via --min-obstacle-area
            clean[labels == i] = 255

    return clean, road_extent


# ----------------------------------------------------------------------
# 2b. OBSTACLE DETECTION -- YOLO (use this on your real machine)
# ----------------------------------------------------------------------
def yolo_obstacle_mask(img_bgr, classes=("car", "truck", "bus")):
    """
    Uses Ultralytics YOLO to detect vehicles and rasterizes their boxes
    into a binary mask. Requires `pip install ultralytics`.
    """
    from ultralytics import YOLO  # local import: optional dependency

    model = YOLO("yolov8n.pt")  # small/fast model, swap for yours
    results = model(img_bgr, verbose=False)[0]

    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    names = results.names
    for box in results.boxes:
        cls_name = names[int(box.cls[0])]
        if cls_name in classes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            mask[y1:y2, x1:x2] = 255
    return mask


# ----------------------------------------------------------------------
# 3. COST GRID
# ----------------------------------------------------------------------
def build_cost_grid(road_mask, obstacle_mask, grid_size=(40, 60)):
    """
    Downsamples road_mask/obstacle_mask into a grid_size = (rows, cols)
    grid of costs:
        0   = free road
        50  = off-road / unknown
        100 = obstacle (blocked)

    Returns: cost_grid (rows, cols) as a numpy array of ints.
    """
    rows, cols = grid_size
    h, w = road_mask.shape

    road_small = cv2.resize(road_mask, (cols, rows), interpolation=cv2.INTER_AREA)
    obs_small = cv2.resize(obstacle_mask, (cols, rows), interpolation=cv2.INTER_AREA)

    cost_grid = np.full((rows, cols), 50, dtype=np.int32)   # default: off-road
    cost_grid[road_small > 127] = 0                          # road = cheap
    cost_grid[obs_small > 60] = 100                           # obstacle = blocked

    return cost_grid


# ----------------------------------------------------------------------
# 4. PATH PLANNING -- A* from left edge to right edge
# ----------------------------------------------------------------------
def astar_left_to_right(cost_grid):
    """
    Finds the cheapest path from ANY cell in the left column to ANY cell
    in the right column, moving 4- or 8-connected, summing cell costs.
    Cells with cost 100 (obstacles) are treated as impassable.

    Returns: list of (row, col) cells from start to goal, or None.
    """
    rows, cols = cost_grid.shape
    BLOCKED = 100

    def neighbors(r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc

    def heuristic(r, c):
        return cols - c  # encourages moving rightward

    # multi-source A*: push every free cell in column 0 as a start.
    # NOTE: we use a counter as a tie-breaker in the heap instead of
    # letting Python fall back to comparing (row, col) tuples directly.
    # Without this, ties (which are common -- many free cells cost the
    # same) always resolve in favor of the smallest row, which is why
    # the path used to hug the top edge of the image no matter what.
    counter = 0
    open_heap = []
    g_score = {}
    came_from = {}

    for r in range(rows):
        if cost_grid[r, 0] < BLOCKED:
            g_score[(r, 0)] = cost_grid[r, 0]
            heapq.heappush(open_heap, (g_score[(r, 0)] + heuristic(r, 0), counter, (r, 0)))
            counter += 1

    visited = set()
    goal_node = None

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        r, c = current
        if c == cols - 1:
            goal_node = current
            break

        for nr, nc in neighbors(r, c):
            if cost_grid[nr, nc] >= BLOCKED:
                continue
            new_g = g_score[current] + 1 + cost_grid[nr, nc]  # step cost + cell cost
            if (nr, nc) not in g_score or new_g < g_score[(nr, nc)]:
                g_score[(nr, nc)] = new_g
                came_from[(nr, nc)] = current
                heapq.heappush(open_heap, (new_g + heuristic(nr, nc), counter, (nr, nc)))
                counter += 1

    if goal_node is None:
        return None  # no path found -- road fully blocked

    # reconstruct path
    path = [goal_node]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


# ----------------------------------------------------------------------
# VISUALIZATION
# ----------------------------------------------------------------------
def render_costmap(img_bgr, road_mask, obstacle_mask, cost_grid, path):
    """
    Overlays: road=green, obstacles=red, path=blue line.
    """
    out = img_bgr.copy()
    overlay = np.zeros_like(img_bgr)
    overlay[road_mask > 0] = (0, 255, 0)        # green = road  (BGR)
    overlay[obstacle_mask > 0] = (0, 0, 255)     # red   = obstacle
    out = cv2.addWeighted(out, 0.6, overlay, 0.4, 0)

    if path:
        rows, cols = cost_grid.shape
        h, w = img_bgr.shape[:2]
        cell_h, cell_w = h / rows, w / cols

        pts = []
        for (r, c) in path:
            x = int((c + 0.5) * cell_w)
            y = int((r + 0.5) * cell_h)
            pts.append((x, y))

        for i in range(len(pts) - 1):
            cv2.line(out, pts[i], pts[i + 1], (255, 0, 0), 4)  # blue line
    else:
        cv2.putText(out, "NO PATH FOUND", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return out


# ----------------------------------------------------------------------
# DEMO IMAGE (used only with --demo, so you can see it run immediately)
# ----------------------------------------------------------------------
def make_demo_image(w=640, h=480):
    img = np.full((h, w, 3), (60, 110, 60), dtype=np.uint8)  # green "grass"
    road_top, road_bot = h // 3, 2 * h // 3
    cv2.rectangle(img, (0, road_top), (w, road_bot), (90, 90, 90), -1)  # gray road
    # a "car" blocking most of the road, leaving a gap at the bottom
    cv2.rectangle(img, (250, road_top + 5), (320, road_bot - 35), (30, 30, 200), -1)
    # a second car further along, blocking the bottom, leaving a gap at the top
    cv2.rectangle(img, (420, road_top + 35), (490, road_bot - 5), (200, 30, 30), -1)
    return img


def save_debug_outputs(out_prefix, img, road_mask, obstacle_mask, road_extent, cost_grid):
    """
    Saves every intermediate stage as its own image so you can see
    exactly why the final result looks the way it does:
        {prefix}_1_road_mask.png      -- raw road classification
        {prefix}_2_road_extent.png    -- convex hull used to scope obstacles
        {prefix}_3_obstacle_mask.png  -- raw obstacle classification
        {prefix}_4_cost_grid.png      -- the actual grid path planning used
    """
    cv2.imwrite(f"{out_prefix}_1_road_mask.png", road_mask)
    cv2.imwrite(f"{out_prefix}_2_road_extent.png", road_extent)
    cv2.imwrite(f"{out_prefix}_3_obstacle_mask.png", obstacle_mask)

    # Cost grid: 0 (road) -> green, 50 (unknown) -> gray, 100 (obstacle) -> red.
    # Upscaled to original image size, with grid lines drawn so you can
    # see the actual cell resolution path planning reasoned over.
    h, w = img.shape[:2]
    rows, cols = cost_grid.shape
    grid_vis = np.zeros((rows, cols, 3), dtype=np.uint8)
    grid_vis[cost_grid == 0] = (0, 200, 0)
    grid_vis[cost_grid == 50] = (90, 90, 90)
    grid_vis[cost_grid == 100] = (0, 0, 220)
    grid_vis = cv2.resize(grid_vis, (w, h), interpolation=cv2.INTER_NEAREST)

    cell_w, cell_h = w / cols, h / rows
    for c in range(1, cols):
        x = int(c * cell_w)
        cv2.line(grid_vis, (x, 0), (x, h), (40, 40, 40), 1)
    for r in range(1, rows):
        y = int(r * cell_h)
        cv2.line(grid_vis, (0, y), (w, y), (40, 40, 40), 1)

    cv2.imwrite(f"{out_prefix}_4_cost_grid.png", grid_vis)
    print(f"Debug images saved with prefix: {out_prefix}_*")


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default=None, help="path to top-down road image")
    parser.add_argument("--out", type=str, default="costmap_result.png")
    parser.add_argument("--demo", action="store_true", help="use a generated test image")
    parser.add_argument("--use-yolo", action="store_true", help="use YOLO instead of classical obstacle detector")
    parser.add_argument("--grid-rows", type=int, default=40)
    parser.add_argument("--grid-cols", type=int, default=60)
    parser.add_argument("--min-obstacle-area", type=int, default=150,
                         help="ignore obstacle blobs smaller than this many pixels (raises this = fewer false-positive obstacles)")
    parser.add_argument("--debug", action="store_true",
                         help="also save road_mask / road_extent / obstacle_mask / cost_grid as separate images")
    args = parser.parse_args()

    if args.demo or args.image is None:
        img = make_demo_image()
    else:
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {args.image}")

    road_mask = segment_road(img)

    if args.use_yolo:
        obstacle_mask = yolo_obstacle_mask(img)
        # YOLO path doesn't compute a road_extent hull -- reuse road_mask
        # itself just so debug output has something sensible to show.
        road_extent = road_mask
    else:
        obstacle_mask, road_extent = classical_obstacle_mask(
            img, road_mask, min_obstacle_area=args.min_obstacle_area)

    cost_grid = build_cost_grid(road_mask, obstacle_mask,
                                 grid_size=(args.grid_rows, args.grid_cols))

    path = astar_left_to_right(cost_grid)

    result = render_costmap(img, road_mask, obstacle_mask, cost_grid, path)

    ok = cv2.imwrite(args.out, result)
    if not ok:
        raise IOError(f"Failed to write {args.out} -- does the parent folder exist?")
    print(f"Saved result to {args.out}")
    print(f"Path found: {'yes' if path else 'no'}")

    if args.debug:
        out_prefix = args.out.rsplit(".", 1)[0]
        save_debug_outputs(out_prefix, img, road_mask, obstacle_mask, road_extent, cost_grid)


if __name__ == "__main__":
    main()