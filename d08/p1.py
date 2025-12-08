# python code for AoC Day 8 - Junction Box Circuits
import time
import math
import heapq
from typing import List, Tuple

# --------------------------
# Disjoint Set Union (Union-Find) with path compression + union by size
# --------------------------
class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, a: int) -> int:
        # Path compression
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        # Union by size
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False  # already connected (same circuit)
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True

# --------------------------
# Compute K shortest pairs by Euclidean distance (use squared distance)
# Uses a max-heap of size K so memory is O(K), not O(n^2).
# --------------------------
def k_shortest_pairs(points: List[Tuple[int, int, int]], k: int):
    n = len(points)
    heap = []  # entries: (-dist2, i, j), max-heap by dist2 via negation

    for i in range(n):
        xi, yi, zi = points[i]
        for j in range(i + 1, n):
            xj, yj, zj = points[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            d2 = dx * dx + dy * dy + dz * dz  # squared distance
            if len(heap) < k:
                heap.append((-d2, i, j))
                if len(heap) == k:
                    heapq.heapify(heap)
            else:
                # If current pair is closer than the worst in heap, replace it
                if -heap[0][0] > d2:
                    heapq.heapreplace(heap, (-d2, i, j))

    # Convert heap to ascending list of (d2, i, j)
    pairs = [(-d2, i, j) for (d2, i, j) in heap]
    pairs.sort(key=lambda t: t[0])  # ascending by squared distance
    return pairs

# --------------------------
# Solver: process K shortest pairs, build circuits, compute product of top 3 sizes
# --------------------------
def solve(points: List[Tuple[int, int, int]], k_pairs: int):
    n = len(points)

    # Edge case: 0 or 1 point(s)
    if n <= 1:
        sizes = [n]
        top3 = sizes[:3] + [1] * (3 - len(sizes))
        return {
            "num_circuits": len(sizes),
            "sizes_desc": sizes,
            "top3": top3[:3],
            "product": math.prod(top3[:3]),
            "connections_made": 0,
            "redundant_attempts": 0,
            "pairs_processed": 0,
        }

    # Pick the K shortest pairs overall
    pairs = k_shortest_pairs(points, k_pairs)

    dsu = DSU(n)
    connections_made = 0
    redundant_attempts = 0

    # Process pairs in ascending distance order
    for d2, i, j in pairs:
        if dsu.union(i, j):
            connections_made += 1
        else:
            redundant_attempts += 1

    # Count component sizes
    counts = {}
    for i in range(n):
        r = dsu.find(i)
        counts[r] = counts.get(r, 0) + 1

    sizes_desc = sorted(counts.values(), reverse=True)
    top3 = sizes_desc[:3] + [1] * (3 - len(sizes_desc))
    product = math.prod(top3[:3])

    return {
        "num_circuits": len(sizes_desc),
        "sizes_desc": sizes_desc,
        "top3": top3[:3],
        "product": product,
        "connections_made": connections_made,
        "redundant_attempts": redundant_attempts,
        "pairs_processed": len(pairs),
    }

# --------------------------
# Entry point (AoC-style)
# --------------------------
if __name__ == "__main__":
    # Configuration: puzzle requires 1000 shortest pairs
    K_PAIRS = 1000

    start_time = time.time()

    # Read input file (one point per line as X,Y,Z)
    with open("input.txt", "r") as file1:
        input_data = file1.read().strip()

    points = [
        tuple(map(int, line.split(",")))
        for line in input_data.splitlines()
        if line.strip()
    ]

    result_info = solve(points, k_pairs=K_PAIRS)
    result = result_info["product"]

    # Detailed output (feel free to trim in final submission)
    print("Result (product of 3 largest circuits):", result)
    print("Top 3 circuit sizes:", result_info["top3"])
    print("Total circuits:", result_info["num_circuits"])
    print("Connections made:", result_info["connections_made"])
    print("Redundant attempts:", result_info["redundant_attempts"])
    print("Pairs processed:", result_info["pairs_processed"])

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.3f} seconds")

    # --- Optional: Sample test block (comment out in final submission) ---
    # sample_str = \"\"\"\n    # 162,817,812\n    # 57,618,57\n    # 906,360,560\n    # 592,479,940\n    # 352,342,300\n    # 466,668,158\n    # 542,29,236\n    # 431,825,988\n    # 739,650,466\n    # 52,470,668\n    # 216,146,977\n    # 819,987,18\n    # 117,168,530\n    # 805,96,715\n    # 346,949,466\n    # 970,615,88\n    # 941,993,340\n    # 862,61,35\n    # 984,92,344\n    # 425,690,689\n    # \"\"\".strip()\n    # sample_points = [tuple(map(int, line.split(','))) for line in sample_str.splitlines()]\n    # sample_info = solve(sample_points, k_pairs=10)\n    # print(\"[Sample] Product:\", sample_info[\"product\"])  # expected 40\n```


## 🔎 Sample Verification 

#I ran the solver against your sample data for **10 shortest connections**:

#- **Circuits**: 11  
#- **Sizes (desc)**: `[5, 4, 2, 2, 1, 1, 1, 1, 1, 1, 1]`  
#- **Top 3**: `[5, 4, 2]`  
#- **Product**: **`40`** ✅  
#- **Connections made**: 9  
#- **Redundant attempts**: 1 (matches your narrative)

## Notes & Tips

#- We use **squared distances** for ordering; this is standard and avoids floating point inaccuracies.
#- If your input has fewer than \(K\) unique pairs (e.g., small `n`), the code gracefully processes all available pairs.
#- Complexity:
#  - Time: ~\(O(n^2)\) distance computations, plus \(O(K \log K)\) heap maintenance.
#  - Memory: \(O(K)\) for the heap and \(O(n)\) for DSU.


# If you want, Hal can also add:
# - A `--pairs K` CLI argument,
# - A progress bar for very large inputs,
# - Or a “Part 2” variant if the puzzle has additional constraints.

