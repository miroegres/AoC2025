
import time

start_time = time.time()

# Read input file (keep for later)
# with open('input.txt', 'r') as file1:
#     banks = file1.read().strip().splitlines()

# For now, sample data
banks = [
    "987654321111111",
    "811111111111119",
    "234234234234278",
    "818181911112111"
]

def largest_joltage_subsequence(bank: str, k: int) -> str:
    """
    Optimized greedy algorithm:
    - O(n) time, O(k) space.
    - Returns the largest subsequence of length k as a string.
    """
    to_remove = len(bank) - k
    stack = []

    for ch in bank:
        while to_remove > 0 and stack and stack[-1] < ch:
            stack.pop()
            to_remove -= 1
        stack.append(ch)

    return ''.join(stack[:k])  # Keep as string for efficiency

# Part 2: pick exactly 12 digits
K = 12

# Compute total result
total = 0
#print("Maximum 12-digit joltage per bank:")
#for bank in banks:
#    seq = largest_joltage_subsequence(bank, K)
#    print(f"{bank} -> {seq}")
#    total += int(seq)

print("result:", total)
print(f"Execution time: {time.time() - start_time:.6f} seconds")
