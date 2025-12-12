
# Advent of Code - Present Packing (Optimized & Verbose)
# Hal's version: early debug prints, faster placements, cached legal options
import time
import re
from functools import lru_cache
from typing import List, Tuple, Set, Dict

# ---------- Parsing ----------

def parse_input(input_data: str) -> Tuple[List[Set[Tuple[int,int]]], List[Tuple[int,int,List[int]]]]:
    """
    Parses shapes and regions from the puzzle input.
    Shapes section:
        index:
        ###..
        #.#..
        ...
    Regions section:
        WxH: c0 c1 c2 ...
    Returns:
        shapes: list indexed by shape id, each is a set of (x,y) coords of '#' cells
        regions: list of (W, H, counts)
    """
    lines = [ln.rstrip() for ln in input_data.splitlines()]
    shapes: Dict[int, List[str]] = {}
    i = 0
    n = len(lines)

    # Helpers
    shape_header_re = re.compile(r'^(\d+):\s*$')
    region_re = re.compile(r'^(\d+)\s*x\s*(\d+)\s*:\s*(.*)$')

    # First pass: read shapes
    current_idx = None
    while i < n:
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue

        m_header = shape_header_re.match(ln)
        m_region = region_re.match(ln)

        if m_region:
            # Regions start; stop shape parsing
            break

        if m_header:
            current_idx = int(m_header.group(1))
            shapes[current_idx] = []
            i += 1
            # Read following rows until blank line or next header/region
            while i < n:
                ln2 = lines[i]
                if not ln2.strip():
                    i += 1
                    break
                if shape_header_re.match(ln2) or region_re.match(ln2):
                    # Next section starts
                    break
                # Expect only '.' and '#'
                if not all(c in ".#" for c in ln2):
                    raise ValueError(f"Invalid shape row for shape {current_idx}: {ln2}")
                shapes[current_idx].append(ln2)
                i += 1
            continue

        # Unexpected non-empty line before regions
        raise ValueError(f"Unexpected line while parsing shapes: {ln}")

    # Normalize shapes into coordinate sets
    if not shapes:
        raise ValueError("No shapes found in input.")

    max_idx = max(shapes.keys())
    # Ensure contiguous indices 0..max_idx
    if set(shapes.keys()) != set(range(max_idx + 1)):
        raise ValueError("Shape indices must be contiguous starting from 0.")

    shape_coords: List[Set[Tuple[int,int]]] = []
    for idx in range(max_idx + 1):
        rows = shapes[idx]
        if not rows:
            raise ValueError(f"Shape {idx} has no rows.")
        width = max(len(r) for r in rows)
        # Pad rows if ragged (rare, but safe)
        rows = [r + '.' * (width - len(r)) for r in rows]
        coords = {(x, y) for y, row in enumerate(rows) for x, ch in enumerate(row) if ch == '#'}
        if not coords:
            raise ValueError(f"Shape {idx} has no '#' cells.")
        # Normalize to origin (0,0)
        minx = min(x for x, y in coords)
        miny = min(y for x, y in coords)
        norm = {(x - minx, y - miny) for (x, y) in coords}
        shape_coords.append(norm)

    # Second pass: read regions
    regions: List[Tuple[int,int,List[int]]] = []
    while i < n:
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        m_region = region_re.match(ln)
        if not m_region:
            raise ValueError(f"Unexpected line while parsing regions: {ln}")
        W = int(m_region.group(1))
        H = int(m_region.group(2))
        counts_str = m_region.group(3).strip()
        if not counts_str:
            raise ValueError(f"Missing counts in region line: {ln}")
        counts = [int(tok) for tok in counts_str.split()]
        if len(counts) != len(shape_coords):
            raise ValueError(f"Region counts length ({len(counts)}) does not match number of shapes ({len(shape_coords)}). Line: {ln}")
        regions.append((W, H, counts))
        i += 1

    return shape_coords, regions

# ---------- Geometry: orientations & placements ----------

def all_orientations(coords: Set[Tuple[int,int]]) -> List[Set[Tuple[int,int]]]:
    """
    Generate all unique orientations (rotations 0/90/180/270 and horizontal flip)
    Normalize each to origin and remove duplicates.
    """
    def normalize(cs: Set[Tuple[int,int]]) -> Set[Tuple[int,int]]:
        minx = min(x for x, y in cs)
        miny = min(y for x, y in cs)
        return {(x - minx, y - miny) for (x, y) in cs}

    def rot90(cs: Set[Tuple[int,int]]) -> Set[Tuple[int,int]]:
        # (x, y) -> (y, -x)
        return {(y, -x) for (x, y) in cs}

    def flipx(cs: Set[Tuple[int,int]]) -> Set[Tuple[int,int]]:
        # horizontal mirror: (x, y) -> (-x, y)
        return {(-x, y) for (x, y) in cs}

    variants = set()
    base = coords
    for flipped in [False, True]:
        cs0 = base if not flipped else flipx(base)
        cs = cs0
        for r in range(4):
            if r > 0:
                cs = rot90(cs)
            norm = normalize(cs)
            variants.add(frozenset(norm))
    return [set(v) for v in variants]

def bounding_box(cs: Set[Tuple[int,int]]) -> Tuple[int,int]:
    maxx = max(x for x, y in cs)
    maxy = max(y for x, y in cs)
    return maxx + 1, maxy + 1  # width, height

def build_placements_for_region(W: int, H: int, shapes_orients: List[List[Set[Tuple[int,int]]]]) -> List[List[Tuple[int, List[int]]]]:
    """
    For a given region size, precompute all placements for each shape.
    Returns placements_by_shape: list (per shape) of list of (mask, cells_list).
    Bit index is y*W + x; occupancy mask uses bits set for '#' cells.
    """
    placements_by_shape: List[List[Tuple[int, List[int]]]] = []
    for s_orients in shapes_orients:
        placements = []
        seen_masks = set()
        for orient in s_orients:
            ow, oh = bounding_box(orient)
            if ow > W or oh > H:
                continue
            for oy in range(H - oh + 1):
                for ox in range(W - ow + 1):
                    mask = 0
                    cells: List[int] = []
                    for (x, y) in orient:
                        idx = (oy + y) * W + (ox + x)
                        mask |= (1 << idx)
                        cells.append(idx)
                    if mask not in seen_masks:
                        seen_masks.add(mask)
                        placements.append((mask, cells))
        placements_by_shape.append(placements)
    return placements_by_shape

# ---------- Global placement cache by board size ----------

PLACEMENTS_CACHE: Dict[Tuple[int,int,Tuple[Tuple[int,int], ...]], List[List[Tuple[int,List[int]]]]] = {}

def placements_for(W: int, H: int, shapes: List[Set[Tuple[int,int]]], debug: bool=False, region_idx: int=0) -> List[List[Tuple[int,List[int]]]]:
    """
    Cache across regions with same W,H and same shapes.
    """
    shapes_signature = tuple(sorted(tuple(sorted(s)) for s in shapes))
    key = (W, H, shapes_signature)
    if key in PLACEMENTS_CACHE:
        return PLACEMENTS_CACHE[key]
    if debug:
        print(f"  [tree {region_idx}] precomputing placements for {W}x{H} ...")
    t0 = time.time()
    shapes_orients = [all_orientations(s) for s in shapes]
    placements = build_placements_for_region(W, H, shapes_orients)
    PLACEMENTS_CACHE[key] = placements
    if debug:
        elapsed = time.time() - t0
        totals = [len(p) for p in placements]
        print(f"  [tree {region_idx}] placements built in {elapsed:.3f}s; per-shape counts: {totals}")
    return placements

# ---------- Solver ----------

def can_fit_region(W: int, H: int, counts: List[int], shapes: List[Set[Tuple[int,int]]], debug: bool=False, region_idx: int=0) -> bool:
    """
    Decide if all presents specified by counts can be placed in W x H without overlap.
    Empty cells are allowed; only non-overlap and on-grid required.
    """
    areas = [len(s) for s in shapes]
    total_area = sum(a * c for a, c in zip(areas, counts))
    if total_area > W * H:
        if debug:
            print(f"  [tree {region_idx}] prune: required area {total_area} > board {W*H}")
        return False

    placements_by_shape = placements_for(W, H, shapes, debug=debug, region_idx=region_idx)

    # If any required shape has zero possible placements, impossible
    for idx, c in enumerate(counts):
        if c > 0 and len(placements_by_shape[idx]) == 0:
            if debug:
                print(f"  [tree {region_idx}] prune: shape {idx} has no placements")
            return False

    # Order shapes (MRV first, then larger areas)
    shape_order = sorted(
        [i for i, c in enumerate(counts) if c > 0],
        key=lambda i: (len(placements_by_shape[i]), -areas[i])
    )

    initial_counts = tuple(counts)
    t0 = time.time()
    nodes = {'count': 0}

    # Cache legal options per (shape, occupied_mask)
    @lru_cache(maxsize=None)
    def legal_options(si: int, occupied_mask: int) -> Tuple[Tuple[int, Tuple[int, ...]], ...]:
        opts = []
        for (mask, cells) in placements_by_shape[si]:
            if (mask & occupied_mask) == 0:
                # store as (mask, tuple(cells)) to be cacheable
                opts.append((mask, tuple(cells)))
        return tuple(opts)

    @lru_cache(maxsize=None)
    def dfs(occupied_mask: int, remaining: Tuple[int, ...]) -> bool:
        nodes['count'] += 1

        # Frequent progress prints
        if debug and nodes['count'] % 10_000 == 0:
            elapsed = time.time() - t0
            print(f"    [tree {region_idx}] nodes={nodes['count']:,} elapsed={elapsed:.2f}s "
                  f"remaining={list(remaining)} occ_bits={occupied_mask.bit_count()}")

        # Success condition
        if all(c == 0 for c in remaining):
            return True

        # Build candidate shapes (with remaining > 0)
        candidate_shapes = [i for i in shape_order if remaining[i] > 0]
        if not candidate_shapes:
            return True

        # Compute legal options and pick most constrained shape
        legal_opts_by_shape: Dict[int, Tuple[Tuple[int, Tuple[int, ...]], ...]] = {}
        pick_si = None
        min_len = None

        for si in candidate_shapes:
            opts = legal_options(si, occupied_mask)
            if len(opts) == 0:
                return False
            legal_opts_by_shape[si] = opts
            if min_len is None or len(opts) < min_len:
                min_len = len(opts)
                pick_si = si

        # Union-capacity prune
        union_mask = 0
        for si in candidate_shapes:
            for (mask, _) in legal_opts_by_shape[si]:
                union_mask |= mask
        remaining_area = sum(areas[i] * remaining[i] for i in range(len(remaining)))
        if union_mask.bit_count() < remaining_area:
            return False

        # Scarcity heuristic: count availability per cell
        cell_avail: Dict[int, int] = {}
        full_mask = (1 << (W * H)) - 1
        free_mask = (~occupied_mask) & full_mask
        # Only count free cells
        for si in candidate_shapes:
            for (mask, cells) in legal_opts_by_shape[si]:
                # Quick skip if placement covers any occupied bit (shouldn't happen)
                if (mask & occupied_mask) != 0:
                    continue
                for idx in cells:
                    if (free_mask >> idx) & 1:
                        cell_avail[idx] = cell_avail.get(idx, 0) + 1

        # Sort options for chosen shape by scarcity score
        opts = legal_opts_by_shape[pick_si]
        def scarcity_score(cell_tuple: Tuple[int, ...]) -> int:
            return sum(cell_avail.get(idx, 0) for idx in cell_tuple)

        opts_sorted = sorted(opts, key=lambda t: (scarcity_score(t[1]), len(t[1])))

        # Try placing one instance of pick_si
        for (mask, cells) in opts_sorted:
            new_occ = occupied_mask | mask
            new_remaining = list(remaining)
            new_remaining[pick_si] -= 1
            if dfs(new_occ, tuple(new_remaining)):
                return True

        return False

    res = dfs(0, initial_counts)
    if debug:
        elapsed = time.time() - t0
        print(f"  [tree {region_idx}] search finished: nodes={nodes['count']:,} time={elapsed:.3f}s result={res}")
    return res

# ---------- Main ----------

if __name__ == "__main__":
    import sys

    # Flags
    DEBUG = ("--debug" in sys.argv)

    # Read input file
    with open('input.txt', 'r') as file1:
        input_data = file1.read().strip()

    start_time = time.time()

    # Parse input
    shapes, regions = parse_input(input_data)

    # Compute how many regions can fit all of their presents
    fit_count = 0
    for idx, (W, H, counts) in enumerate(regions, start=1):
        if DEBUG:
            print(f"[tree {idx}] board={W}x{H} counts={counts}")
        ok = can_fit_region(W, H, counts, shapes, debug=DEBUG, region_idx=idx)
        if ok:
            fit_count += 1
        if DEBUG:
            print(f"[tree {idx}] {'OK' if ok else 'FAIL'}")
            print("-" * 50)

    result = fit_count

    execution_time = time.time() - start_time
    print("result:", result)
   
    print(f"execution time: {execution_time:.3f}s")