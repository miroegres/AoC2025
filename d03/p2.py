
# python code for AoC
import re
import time

start_time = time.time()

# Read input file (keep this for later use)
with open('input.txt', 'r') as file1:
    input_data = file1.read().strip()  # Read entire file and remove extra spaces/newlines

# For now, use sample data
#input_data = """987654321111111
#811111111111119
#234234234234278
#818181911112111"""

# Split into banks (each line is a bank)
banks = input_data.splitlines()

def largest_joltage_subsequence(bank: str, k: int) -> int:
    """
    Return the largest possible joltage formed by picking exactly k digits
    from 'bank' while preserving their original order.

    Greedy stack algorithm:
    - Let 'to_remove' = len(bank) - k (the number of digits we must drop).
    - Scan digits; while we can remove and the last chosen digit is smaller
      than the current digit, pop it to make the number larger.
    - Append current digit.
    - After the scan, take the first k digits of the stack.
    """
    n = len(bank)
    if k > n:
        raise ValueError(f"Bank length {n} is smaller than required k={k}")

    to_remove = n - k
    stack = []

    for ch in bank:
        while to_remove > 0 and stack and stack[-1] < ch:
            stack.pop()
            to_remove -= 1
        stack.append(ch)

    # If we still need to remove, drop from the end
    result_digits = stack[:k]
    return int(''.join(result_digits))

# Part 2: pick exactly 12 digits
K = 12

# Compute per-bank maximum joltage and total
per_bank = []
for bank in banks:
    per_bank.append(largest_joltage_subsequence(bank, K))
result = sum(per_bank)

# Print details
#print("Maximum 12-digit joltage per bank:")
#for bank, val in zip(banks, per_bank):
#    print(f"{bank} -> {val}")

print("result:", result)

execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.6f} seconds")
