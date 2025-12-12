
# Advent of Code - Present Packing (Cell-Guided + Parallel + Verbose)
# Hal's version: branch on most constrained free cell, plus multiprocessing (Windows-safe)
import time
import re
import os
import sys
from functools import lru_cache
from typing import List, Tuple, Set, Dict, Optional

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
        shapes: list indexed by shape id, each is a set of (x,y) coords of '#' cells (normalized to origin)
        regions: list of (W, H, counts)
    """
    lines = [ln.rstrip() for ln in input_data.splitlines()]
    shapes: Dict[int, List[str]] = {}
    i = 0
    n = len(lines)

    shape_header_re = re.compile(r'^(\d+):\s*$')
    region_re = re.compile(r'^(\d+)\s*x\s*(\d+)\s*:\s*(.*)$')

    while i < n:
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue

        m_header = shape_header_re.match(ln)
        m_region = region_re.match(ln)

        if m_region:
            break

        if m_header:
            idx = int(m_header.group(1))
            shapes[idx] = []
            i += 1
            while i < n:
                ln2 = lines[i]
                if not ln2.strip():
                    i += 1
                    break
                if shape_header_re.match(ln2) or region_re.match(ln2):
                    break
                if not all(c in ".#" for c in ln2):
                    raise ValueError(f"Invalid shape row for shape {idx}: {ln2}")
                shapes[idx].append(ln2)
                i += 1
            continue

        raise ValueError(f"Unexpected line while parsing shapes: {ln}")

    if not shapes:
        raise ValueError("No shapes found in input.")

    max_idx = max(shapes.keys())
    if set(shapes.keys()) != set(range(max_idx + 1)):
        raise ValueError("Shape indices must be contiguous starting from 0.")

    shape_coords: List[Set[Tuple[int,int]]] = []
    for idx in range(max_idx + 1):
        rows = shapes[idx]
        width = max(len(r) for r in rows)
        rows = [r + '.' * (width - len(r)) for r in rows]
        coords = {(x, y) for y, row in enumerate(rows) for x, ch in enumerate(row) if ch == '#'}
        if not coords:
            raise ValueError(f"Shape {idx} has no '#' cells.")
        minx = min(x for x, y in coords)
        miny = min(y for x, y in coords)
        norm = {(x - minx, y - miny) for (x, y) in coords}
        shape_coords.append(norm)

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
        counts = [int(tok) for tok in counts_str.split()]
        if len(counts) != len(shape_coords):
            raise ValueError(f"Region counts length ({len(counts)}) does not match number of shapes ({len(shape_coords)}). Line: {ln}")
        regions.append((W, H, counts))
        i += 1

    return shape_coords, regions

# ---------- Geometry: orientations & placements ----------

def all_orientations(coords: Set[Tuple[int,int]]) -> List[Set[Tuple[int,int]]]:
    """
    Generate all unique orientations (rotations 0/90/180/270 and horizontal flip).
    Normalize each to origin and remove duplicates.
    """
    def normalize(cs: Set[Tuple[int,int]]) -> Set[Tuple[int,int]]:
        minx = min(x for x, y in cs)
        miny = min(y for x, y in cs)
        return {(x - minx, y - miny) for (x, y) in cs}

    def rot90(cs: Set[Tuple[int,int]]) -> Set[Tuple[int,int]]:
        return {(y, -x) for (x, y) in cs}

    def flipx(cs: Set[Tuple[int,int]]) -> Set[Tuple[int,int]]:
        return {(-x, y) for (x, y) in cs}

    variants = set()
    base = coords
    for flipped in [False, True]:
        cs = base if not flipped else flipx(base)
        for r in range(4):
            if r > 0:
                cs = rot90(cs)
            variants.add(frozenset(normalize(cs)))
    return [set(v) for v in variants]

def bounding_box(cs: Set[Tuple[int,int]]) -> Tuple[int,int]:
    maxx = max(x for x, y in cs)
    maxy = max(y for x, y in cs)
    return maxx + 1, maxy + 1

def build_placements_for_region(W: int, H: int, shapes_orients: List[List[Set[Tuple[int,int]]]]) -> List[List[Tuple[int, List[int]]]]:
    """
    Precompute all placements for each shape:
    returns placements_by_shape[si] = list of (mask, cells_list)
    mask bit index: y*W + x
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

def build_cover_index(W: int, H: int, placements_by_shape: List[List[Tuple[int, List[int]]]]) -> List[List[Tuple[int, int, Tuple[int, ...]]]]:
    """
    For each cell, list all placements (shape idx, mask, cells_tuple) covering it.
    """
    cover: List[List[Tuple[int, int, Tuple[int, ...]]]] = [[] for _ in range(W * H)]
    for si, plist in enumerate(placements_by_shape):
        for (mask, cells) in plist:
            cell_tuple = tuple(cells)
            for idx in cell_tuple:
                cover[idx].append((si, mask, cell_tuple))
    return cover

# ---------- Global caches ----------

PLACEMENTS_CACHE: Dict[Tuple[int,int,Tuple[Tuple[int,int], ...]], List[List[Tuple[int,List[int]]]]] = {}
COVER_CACHE: Dict[Tuple[int,int,Tuple[Tuple[int,int], ...]], List[List[Tuple[int,int,Tuple[int, ...]]]]] = {}

def placements_and_cover_for(W: int, H: int, shapes: List[Set[Tuple[int,int]]], debug: bool=False, region_idx: int=0):
    shapes_signature = tuple(sorted(tuple(sorted(s)) for s in shapes))
    key = (W, H, shapes_signature)
    if key in PLACEMENTS_CACHE and key in COVER_CACHE:
        return PLACEMENTS_CACHE[key], COVER_CACHE[key]
    if debug:
        print(f"  [tree {region_idx}] precomputing placements+cover for {W}x{H} ...")
    t0 = time.time()
    shapes_orients = [all_orientations(s) for s in shapes]
    placements = build_placements_for_region(W, H, shapes_orients)
    cover = build_cover_index(W, H, placements)
    PLACEMENTS_CACHE[key] = placements
    COVER_CACHE[key] = cover
    if debug:
        elapsed = time.time() - t0
        totals = [len(p) for p in placements]
        print(f"  [tree {region_idx}] placements built in {elapsed:.3f}s; per-shape counts: {totals}")
    return placements, cover

# ---------- Solver (cell-guided) ----------

def can_fit_region(
    W: int,
    H: int,
    counts: List[int],
    shapes: List[Set[Tuple[int,int]]],
    debug: bool=False,
    region_idx: int=0,
    timeout: Optional[float]=None,
    progress_interval: int = 5000
) -> bool:
    """
    Cell-guided DFS:
    - If all required shapes placed -> success (empty cells allowed).
    - At each step, pick the free cell that has the fewest legal placements covering it,
      across all remaining shapes; branch on those placements.
    """
    areas = [len(s) for s in shapes]
    total_area = sum(a * c for a, c in zip(areas, counts))
    if total_area > W * H:
        if debug:
            print(f"  [tree {region_idx}] prune: required area {total_area} > board {W*H}")
        return False

    placements_by_shape, cover_index = placements_and_cover_for(W, H, shapes, debug=debug, region_idx=region_idx)

    # If any required shape has zero possible placements, impossible
    for idx, c in enumerate(counts):
        if c > 0 and len(placements_by_shape[idx]) == 0:
            if debug:
                print(f"  [tree {region_idx}] prune: shape {idx} has no placements")
            return False

    initial_counts = tuple(counts)
    t0 = time.time()
    nodes = {'count': 0}

    def union_legal_mask(occupied_mask: int, remaining: Tuple[int, ...]) -> int:
        u = 0
        for si, rem in enumerate(remaining):
            if rem <= 0:
                continue
            for (mask, _) in placements_by_shape[si]:
                if (mask & occupied_mask) == 0:
                    u |= mask
        return u

    @lru_cache(maxsize=None)
    def dfs(occupied_mask: int, remaining: Tuple[int, ...]) -> bool:
        # Timeout
        if timeout is not None and (time.time() - t0) > timeout:
            return False

        nodes['count'] += 1
        if debug and nodes['count'] % progress_interval == 0:
            elapsed = time.time() - t0
            print(f"    [tree {region_idx}] nodes={nodes['count']:,} elapsed={elapsed:.2f}s "
                  f"remaining={list(remaining)} occ_bits={occupied_mask.bit_count()}")

        # Success: all shapes placed
        if all(c == 0 for c in remaining):
            return True

        # Union capacity prune
        union_mask = union_legal_mask(occupied_mask, remaining)
        remaining_area = sum(areas[i] * remaining[i] for i in range(len(remaining)))
        if union_mask == 0 or union_mask.bit_count() < remaining_area:
            return False

        # Pick the most constrained free cell (among cells that have at least one legal placement)
        full_mask = (1 << (W * H)) - 1
        free_mask = (~occupied_mask) & full_mask
        candidate_cells = union_mask & free_mask
        if candidate_cells == 0:
            # No free cell can be covered by any remaining placement -> fail
            return False

        # Iterate set bits of candidate_cells and find cell with minimal legal options
        min_cell = None
        min_options: List[Tuple[int,int,Tuple[int, ...]]] = None
        min_len = None

        cm = candidate_cells
        while cm:
            lsb = cm & -cm
            cell = (lsb.bit_length() - 1)
            cm ^= lsb

            # Build legal options that cover this cell
            opts_here: List[Tuple[int,int,Tuple[int, ...]]] = []
            for (si, mask, cells_tuple) in cover_index[cell]:
                if remaining[si] <= 0:
                    continue
                if (mask & occupied_mask) == 0:
                    opts_here.append((si, mask, cells_tuple))
            if opts_here:
                l = len(opts_here)
                if min_len is None or l < min_len:
                    min_len = l
                    min_cell = cell
                    min_options = opts_here
                    # Early MRV cutoff: if 1 option, that's best possible
                    if min_len == 1:
                        break

        if min_options is None or len(min_options) == 0:
            return False

        # Optional ordering: try larger shapes first (cover more area)
        def opt_key(item: Tuple[int,int,Tuple[int, ...]]):
            si, mask, cells_tuple = item
            return (-areas[si], len(cells_tuple))
        min_options.sort(key=opt_key)

        # Branch on each placement that covers min_cell
        for (si, mask, cells_tuple) in min_options:
            new_occ = occupied_mask | mask
            new_remaining = list(remaining)
            new_remaining[si] -= 1
            if dfs(new_occ, tuple(new_remaining)):
                return True

        return False

    res = dfs(0, initial_counts)
    if debug:
        elapsed = time.time() - t0
        print(f"  [tree {region_idx}] search finished: nodes={nodes['count']:,} time={elapsed:.3f}s result={res}")
    return res

# ---------- Multiprocessing Worker (top-level, Windows-safe) ----------

def solve_one_tree(args: Tuple[int, int, int, List[int], List[Set[Tuple[int,int]]], Optional[float], int]) -> Tuple[int, bool]:
    """
    Top-level worker for ProcessPool    Top-level worker for ProcessPoolExecutor (must be pickleable on Windows).
    Args: (idx, W, H, counts, shapes, timeout, progress_interval)
    Returns: (idx, ok)
    """
    idx, W, H, counts, shapes, timeout, progress_interval = args
    ok = can_fit_region(W, H, counts, shapes, debug=False, region_idx=idx, timeout=timeout, progress_interval=progress_interval)
    return idx, ok

# ---------- Main ----------

def main():
    # Flags
    DEBUG = ("--debug" in sys.argv)

    # Optional flags: --timeout=SECONDS, --workers=N, --progress=NODES
    timeout: Optional[float] = None
    workers: Optional[int] = None
    progress_interval: int = 5000

    for a in sys.argv[1:]:
        if a.startswith("--timeout="):
            try:
                timeout = float(a.split("=", 1)[1])
            except:
                pass
        elif a.startswith("--workers="):
            try:
                workers = int(a.split("=", 1)[1])
            except:
                pass
        elif a.startswith("--progress="):
            try:
                progress_interval = int(a.split("=", 1)[1])
            except:
                pass

    with open('input.txt', 'r') as file1:
        input_data = file1.read().strip()

    start_time = time.time()
    shapes, regions = parse_input(input_data)

    # Compute how many regions can fit all of their presents
    results: List[bool] = []

    if DEBUG:
        for idx, (W, H, counts) in enumerate(regions, start=1):
            print(f"[tree {idx}] board={W}x{H} counts={counts}")
            ok = can_fit_region(W, H, counts, shapes, debug=True, region_idx=idx, timeout=timeout, progress_interval=progress_interval)
            results.append(ok)
            print(f"[tree {idx}] {'OK' if ok else 'FAIL'}")
            print("-" * 50)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        tasks = [(i+1, W, H, counts, shapes, timeout, progress_interval) for i, (W, H, counts) in enumerate(regions)]
        max_workers = workers or os.cpu_count() or 1
        print(f"[info] solving {len(tasks)} trees using {max_workers} workers ...")
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(solve_one_tree, t) for t in tasks]
            for fut in as_completed(futures):
                idx, ok = fut.result()
                results.append(ok)

    fit_count = sum(1 for r in results if r)
    execution_time = time.time() - start_time
    print("result:", fit_count)
    print(f"Execution time: {execution_time:.3f} seconds")

if __name__ == "__main__":
    main()