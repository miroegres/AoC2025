
# Advent of Code – Day 9 (Bands-based, multi-core Part 2, with progress & debug render)
import time
from bisect import bisect_left, bisect_right
from multiprocessing import Pool, cpu_count

VERBOSE = True  # set False to reduce logs

def log(msg):
    if VERBOSE:
        print(msg)

# ------------------------ Parsing ------------------------

def parse_input(text):
    points = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ',' in line:
            x_str, y_str = line.split(',', 1)
        else:
            x_str, y_str = line.split()
        points.append((int(x_str), int(y_str)))
    return points

# ------------------------ Segments ------------------------

def build_segments(points):
    """
    Build axis-aligned segments between consecutive red points (wrap).
    Returns:
      vertical_segments: list of (x, ya, yb) with ya < yb
      horizontal_segments: list of (y, xa, xb) with xa < xb
      bbox: (minx, miny, maxx, maxy)
      red_by_row: dict[y] -> set of x (for row-wise red degenerate intervals in validation)
    """
    red_by_row = {}
    for x, y in points:
        red_by_row.setdefault(y, set()).add(x)

    vertical_segments = []
    horizontal_segments = []
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        if x1 == x2:
            ya, yb = sorted((y1, y2))
            vertical_segments.append((x1, ya, yb))
        elif y1 == y2:
            xa, xb = sorted((x1, x2))
            horizontal_segments.append((y1, xa, xb))
        else:
            raise ValueError("Input path must be axis-aligned")

    # bbox from points and segments
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    minx = min(xs); maxx = max(xs)
    miny = min(ys); maxy = max(ys)
    for y, xa, xb in horizontal_segments:
        minx = min(minx, xa)
        maxx = max(maxx, xb)
        miny = min(miny, y)
        maxy = max(maxy, y)
    for x, ya, yb in vertical_segments:
        minx = min(minx, x)
        maxx = max(maxx, x)
        miny = min(miny, ya)
        maxy = max(maxy, yb)

    bbox = (minx, miny, maxx, maxy)
    return vertical_segments, horizontal_segments, bbox, red_by_row

# ------------------------ Row intervals ------------------------

def merge_intervals(intervals):
    """Merge sorted intervals [(L, R)] into disjoint list."""
    if not intervals:
        return []
    merged = []
    curL, curR = intervals[0]
    for L, R in intervals[1:]:
        if L <= curR + 1:
            curR = max(curR, R)
        else:
            merged.append((curL, curR))
            curL, curR = L, R
    merged.append((curL, curR))
    return merged

def build_horizontal_map(horizontal_segments, minx, maxx):
    """
    For each horizontal row y, build merged intervals (xa+1..xb-1), clamped to bbox.
    Returns: horiz_map: dict[y] -> merged intervals; horiz_ys_sorted: sorted list of ys.
    """
    row_map = {}
    for y, xa, xb in horizontal_segments:
        L = xa + 1
        R = xb - 1
        if L <= R:
            Lc = max(L, minx)
            Rc = min(R, maxx)
            if Lc <= Rc:
                row_map.setdefault(y, []).append((Lc, Rc))
    for y, ints in list(row_map.items()):
        ints.sort()
        row_map[y] = merge_intervals(ints)
    return row_map, sorted(row_map.keys())

# ------------------------ Bands (vertical sweep w/ [ya, yb) rule) ------------------------

def build_bands(vertical_segments, bbox):
    """
    Sweep Y to produce bands where the set of active vertical crossings is constant.
    Use half-open rule: vertical edge active on rows y ∈ [ya, yb).
    Each band's intervals consist of:
      - degenerate intervals for each active vertical edge: (x, x)  [boundary]
      - interior intervals between pairs: (xs[i]+1 .. xs[i+1]-1)
    Returns: bands: list[{'y0': int, 'y1': int, 'intervals': list[(L, R)] (merged)}]
    """
    minx, miny, maxx, maxy = bbox

    start_events = {}  # at ya
    end_events   = {}  # at yb (upper-exclusive)
    for x, ya, yb in vertical_segments:
        start_events.setdefault(ya, []).append(x)
        end_events.setdefault(yb, []).append(x)

    event_ys = set([miny, maxy + 1]) | set(start_events.keys()) | set(end_events.keys())
    event_list = sorted(event_ys)

    active = set()
    bands = []

    for i in range(len(event_list) - 1):
        y0 = event_list[i]
        y1 = event_list[i + 1] - 1
        if y0 < miny:
            continue
        if y1 > maxy:
            y1 = maxy

        # End events processed first so edges ending at y0 are not active in this band
        for x in end_events.get(y0, []):
            if x in active:
                active.remove(x)
        # Start events at y0 are active for this band
        for x in start_events.get(y0, []):
            active.add(x)

        if y0 > y1:
            continue

        xs_sorted = sorted(active)
        intervals = []

        # Degenerate intervals for boundary vertical edges
        for x in xs_sorted:
            if minx <= x <= maxx:
                intervals.append((x, x))

        # Interior intervals between pairs
        for k in range(0, len(xs_sorted) - 1, 2):
            L = xs_sorted[k] + 1
            R = xs_sorted[k + 1] - 1
            if L <= R:
                Lc = max(L, minx); Rc = min(R, maxx)
                if Lc <= Rc:
                    intervals.append((Lc, Rc))

        intervals.sort()
        merged = merge_intervals(intervals)
        bands.append({'y0': y0, 'y1': y1, 'intervals': merged})

    return bands

# ------------------------ Rectangle validation ------------------------

def interval_covers(intervals, L, R):
    """
    Check if union of 'intervals' (merged, sorted) fully covers [L..R].
    Because merged intervals are disjoint, coverage requires a single interval [a,b] with a<=L and b>=R.
    """
    if not intervals:
        return False
    starts = [a for a, _ in intervals]
    idx = bisect_right(starts, L) - 1
    if idx >= 0:
        a, b = intervals[idx]
        if a <= L and b >= R:
            return True
    return False

def count_horiz_rows_in_range(horiz_ys_sorted, yA, yB):
    """Count horizontal rows in [yA..yB] via binary search."""
    lo = bisect_left(horiz_ys_sorted, yA)
    hi = bisect_right(horiz_ys_sorted, yB)
    return hi - lo, lo, hi

def validate_rectangle(x1, y1, x2, y2, bands, horiz_map, horiz_ys_sorted, red_by_row):
    """
    Validate rectangle (x1,y1)-(x2,y2) entirely allowed using:
      - band intervals (boundary verticals + interior)
      - horizontal edge intervals on horizontal rows
      - red points as degenerate intervals on their rows (for endpoints)
    """
    L = min(x1, x2)
    R = max(x1, x2)
    YA = min(y1, y2)
    YB = max(y1, y2)

    for b in bands:
        # Skip bands outside rectangle Y-range
        if b['y1'] < YA or b['y0'] > YB:
            continue

        yS = max(YA, b['y0'])
        yE = min(YB, b['y1'])

        # If the band intervals already cover [L..R], we're good for all rows in this intersection.
        if interval_covers(b['intervals'], L, R):
            continue

        # Otherwise, only acceptable if *every* row in [yS..yE] is a horizontal-edge row,
        # and for each such row, the union (band intervals + horizontal row intervals + red points on that row)
        # covers [L..R].
        cnt, lo, hi = count_horiz_rows_in_range(horiz_ys_sorted, yS, yE)
        total_rows = yE - yS + 1
        non_special = total_rows - cnt
        if non_special > 0:
            return False  # non-horizontal rows lacking full band coverage => fail

        for idx in range(lo, hi):
            y = horiz_ys_sorted[idx]
            base = b['intervals']
            extra = horiz_map.get(y, [])
            # Add red points on this row as degenerate intervals
            red_ints = [(x, x) for x in sorted(red_by_row.get(y, []))]
            # Quick wins first
            if interval_covers(base, L, R) or interval_covers(extra, L, R):
                continue

            # Merge base + extra + red into a union and test coverage
            i = j = k = 0
            lists = [base, extra, red_ints]
            heads = [lists[0][0] if lists[0] else None,
                     lists[1][0] if lists[1] else None,
                     lists[2][0] if lists[2] else None]
            curL = None
            curR = None
            covered = False

            # k-way merge by smallest start
            while True:
                # Select next smallest start
                m_idx = -1
                m_val = None
                for idx2, head in enumerate(heads):
                    if head is None:
                        continue
                    if m_val is None or head[0] < m_val[0]:
                        m_val = head
                        m_idx = idx2
                if m_idx == -1:
                    break
                cand = heads[m_idx]
                # Advance cursor in that list
                if m_idx == 0:
                    i += 1
                    heads[0] = lists[0][i] if i < len(lists[0]) else None
                elif m_idx == 1:
                    j += 1
                    heads[1] = lists[1][j] if j < len(lists[1]) else None
                else:
                    k += 1
                    heads[2] = lists[2][k] if k < len(lists[2]) else None

                # Merge into current union
                if curL is None:
                    curL, curR = cand
                else:
                    if cand[0] <= curR + 1:
                        curR = max(curR, cand[1])
                    else:
                        # finalize previous and start new
                        if curL <= L and curR >= R:
                            covered = True
                            break
                        curL, curR = cand

            if not covered and curL is not None and curL <= L and curR >= R:
                covered = True

            if not covered:
                return False

    return True

# ------------------------ Part 1 / Part 2 ------------------------

def largest_rectangle_area_part1(points):
    max_area = 0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            area = (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)
            if area > max_area:
                max_area = area
    return max_area

# Worker globals
_SH_BANDS = None
_SH_HORIZ_MAP = None
_SH_HORIZ_YS = None
_SH_POINTS = None
_SH_RED_BY_ROW = None

def _init_worker(bands, horiz_map, horiz_ys_sorted, points, red_by_row):
    global _SH_BANDS, _SH_HORIZ_MAP, _SH_HORIZ_YS, _SH_POINTS, _SH_RED_BY_ROW
    _SH_BANDS = bands
    _SH_HORIZ_MAP = horiz_map
    _SH_HORIZ_YS = horiz_ys_sorted
    _SH_POINTS = points
    _SH_RED_BY_ROW = red_by_row

def _worker_part2(i):
    pts = _SH_POINTS
    x1, y1 = pts[i]
    local_max = 0
    for j in range(i + 1, len(pts)):
        x2, y2 = pts[j]
        area = (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)
        if validate_rectangle(x1, y1, x2, y2, _SH_BANDS, _SH_HORIZ_MAP, _SH_HORIZ_YS, _SH_RED_BY_ROW):
            if area > local_max:
                local_max = area
    return local_max

def largest_rectangle_area_part2_parallel(points, bands, horiz_map, horiz_ys_sorted, red_by_row, processes=None):
    if processes is None:
        processes = max(1, cpu_count() or 1)
    log(f"  Launching {processes} workers for Part 2 pair search")
    with Pool(processes=processes, initializer=_init_worker,
              initargs=(bands, horiz_map, horiz_ys_sorted, points, red_by_row)) as pool:
        results = pool.map(_worker_part2, range(len(points)), chunksize=64)
    return max(results) if results else 0

# ------------------------ Debug Render ------------------------

def build_allowed_by_row_from_bands(bands, horiz_map, red_by_row, miny, maxy):
    """
    For small debug rendering: expand bands into per-row intervals, union with horizontal intervals
    and red degenerate points. Not used in computation; only render_grid.
    """
    allowed_by_row = {}
    for b in bands:
        for y in range(b['y0'], b['y1'] + 1):
            allowed_by_row[y] = list(b['intervals'])
    for y, extras in horiz_map.items():
        base = allowed_by_row.get(y, [])
        merged = merge_intervals(sorted(base + extras))
        allowed_by_row[y] = merged
    for y, reds in red_by_row.items():
        base = allowed_by_row.get(y, [])
        red_ints = [(x, x) for x in sorted(reds)]
        merged = merge_intervals(sorted(base + red_ints))
        allowed_by_row[y] = merged
    return allowed_by_row

def render_grid(bbox, red_by_row, bands, horiz_map, highlight=None, max_w=120, max_h=60):
    """
    Renders a clipped ASCII view:
      - '.' empty
      - '#' red
      - 'X' green (boundary + interior)
      - 'O' highlighted rectangle (if provided: ((x1,y1),(x2,y2)))
    """
    minx, miny, maxx, maxy = bbox
    W = maxx - minx + 1
    H = maxy - miny + 1
    if W > max_w or H > max_h:
        print(f"[render_grid] Skipping render: bbox {W}x{H} exceeds {max_w}x{max_h}")
        return

    allowed_by_row = build_allowed_by_row_from_bands(bands, horiz_map, red_by_row, miny, maxy)

    ox1 = oy1 = ox2 = oy2 = None
    if highlight:
        (hx1, hy1), (hx2, hy2) = highlight
        ox1, ox2 = min(hx1, hx2), max(hx1, hx2)
        oy1, oy2 = min(hy1, hy2), max(hy1, hy2)

    lines = []
    for y in range(miny, maxy + 1):
        row_chars = []
        reds = red_by_row.get(y, set())
        intervals = allowed_by_row.get(y, [])
        it = iter(intervals)
        cur = next(it, None)
        for x in range(minx, maxx + 1):
            while cur and x > cur[1]:
                cur = next(it, None)
            allowed = cur and cur[0] <= x <= cur[1]
            ch = 'X' if allowed else '.'
            if x in reds:
                ch = '#'
            if highlight and (ox1 is not None) and (ox1 <= x <= ox2) and (oy1 <= y <= oy2):
                ch = 'O'
            row_chars.append(ch)
        lines.append(''.join(row_chars))
    print("\n".join(lines))

# ------------------------ Main ------------------------

if __name__ == "__main__":
    t_all = time.time()
    with open('input.txt', 'r') as f:
        input_data = f.read().strip()

    points = parse_input(input_data)
    log(f"Loaded {len(points)} red points")

    # PART 1 (fast)
    t1 = time.time()
    part1 = largest_rectangle_area_part1(points)
    print("Part 1:", part1)
    print(f"Part 1 time: {time.time() - t1:.3f} s")

    # Precompute bands and horizontal map
    t_build = time.time()
    vertical_segments, horizontal_segments, bbox, red_by_row = build_segments(points)
    minx, miny, maxx, maxy = bbox
    W = maxx - minx + 1
    H = maxy - miny + 1
    log(f"BBox: {W} x {H} from ({minx},{miny}) to ({maxx},{maxy})")
    log(f"Segments: {len(vertical_segments)} vertical, {len(horizontal_segments)} horizontal")

    horiz_map, horiz_ys_sorted = build_horizontal_map(horizontal_segments, minx, maxx)
    bands = build_bands(vertical_segments, bbox)
    log(f"Bands: {len(bands)}")

    print(f"Precompute (bands + horizontals) time: {time.time() - t_build:.3f} s")

    # PART 2 (multi-core)
    t2 = time.time()
    processes = max(1, cpu_count() or 1)
    part2 = largest_rectangle_area_part2_parallel(points, bands, horiz_map, horiz_ys_sorted, red_by_row, processes=processes)
    print("Part 2:", part2)
    print(f"Part 2 time: {time.time() - t2:.3f} s")
    print(f"Total time: {time.time() - t_all:.3f} s")

    # OPTIONAL: render a small grid (auto-skips if bbox large)
    # render_grid(bbox, red_by_row, bands, horiz_map,
    #             highlight=(points[0], points[-1]),
    #             max_w=120, max_h=60)
