
# python code for AoC Day 7 - Part 2: final connection X-product
import time
from typing import List, Tuple

# --------------------------
# Disjoint Set Union (Union-Find) with path compression + union by size
# --------------------------
class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.components = n  # track # of connected components

    def find(self, a: int) -> int:
        # Path compression
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a: int, b: int) -> bool:
        # Union by size; returns True if a merge happened
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        self.components -= 1
        return True

# --------------------------
# Part 2 solver
# Build all pairs, sort by squared distance, union until one component remains.
# Return last pair that achieves single circuit + X-product.
# --------------------------
def solve_until_single(points: List[Tuple[int, int, int]]):
    n = len(points)
    if n == 0:
        return {"last_pair": None, "product": 0, "steps": 0}
    if n == 1:
        return {"last_pair": None, "product": points[0][0], "steps": 0}

    # Generate ALL pairs with squared distance (O(n^2) pairs)
    pairs = []
    for i in range(n):
        xi, yi, zi = points[i]
        for j in range(i + 1, n):
            xj, yj, zj = points[j]
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj
            d2 = dx * dx + dy * dy + dz * dz
            pairs.append((d2, i, j))

    # Sort ascending by squared distance
    pairs.sort(key=lambda t: t[0])

    # Union pairs in order until single component remains
    dsu = DSU(n)
    last_pair = None
    steps = 0

    for d2, i, j in pairs:
        if dsu.union(i, j):
            steps += 1
            last_pair = (i, j)
            if dsu.components == 1:
                # Final connection achieved
                x_prod = points[i][0] * points[j][0]
                return {
                    "last_pair": last_pair,
                    "product": x_prod,
                    "steps": steps,
                }

    # Shouldn't happen unless input is empty; return best-effort info
    return {"last_pair": last_pair, "product": None, "steps": steps}

# --------------------------
# Entry point (AoC-style)
# --------------------------
if __name__ == "__main__":
    start_time = time.time()

    # Read input file: one point per line as X,Y,Z
    with open("input.txt", "r") as file1:
        input_data = file1.read().strip()

    # Parse points
    points = [
        tuple(map(int, line.split(",")))
        for line in input_data.splitlines()
        if line.strip()
    ]

    info = solve_until_single(points)
    result = info["product"]

    # Output
    print("Result (X-product of final connection):", result)
    if info["last_pair"] is not None:
        i, j = info["last_pair"]
        print("Final pair indices:", (i, j))
        print("Final pair points:", points[i], points[j])
    print("Union steps:", info["steps"])

    execution_time = time.time() - start_time
    print(f"Execution time: {execution_time:.3f} seconds")
