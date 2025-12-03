
# Advent of Code - Battery Joltage Puzzle
# Optimized solution for Part 1 and Part 2 with detailed comments

import time

start_time = time.time()

# Read input file (streaming approach for large files)
# Uncomment for real input:
with open('input.txt', 'r') as file1:
    banks = file1.read().strip().splitlines()

# For now, use sample data
#banks = [
#    "987654321111111",
#    "811111111111119",
#    "234234234234278",
#    "818181911112111"
#]

# -------------------------------
# PART 1: Pick exactly TWO digits for max joltage
# -------------------------------
def max_two_digit_joltage(bank: str) -> int:
    """
    Find the largest possible two-digit number by picking two digits
    in original order (cannot rearrange).
    
    Optimization:
    - Instead of checking all pairs (O(n²)), we:
      1. Track the largest digit seen so far.
      2. For each digit, compute possible two-digit number with previous max.
    - Complexity: O(n)
    """
    max_joltage = 0
    max_left_digit = bank[0]  # First digit as starting point

    for i in range(1, len(bank)):
        # Form two-digit number using max_left_digit and current digit
        joltage = int(max_left_digit + bank[i])
        if joltage > max_joltage:
            max_joltage = joltage

        # Update max_left_digit if current digit is larger
        if bank[i] > max_left_digit:
            max_left_digit = bank[i]

    return max_joltage

# Compute Part 1 result
part1_total = sum(max_two_digit_joltage(bank) for bank in banks)

# -------------------------------
# PART 2: Pick exactly 12 digits for max joltage
# -------------------------------
def largest_joltage_subsequence(bank: str, k: int) -> str:
    """
    Return the largest possible number formed by picking exactly k digits
    from 'bank' while preserving order.
    
    Approach:
    - Greedy stack algorithm:
      * We must remove (len(bank) - k) digits.
      * While we can remove and the last digit in stack is smaller than current,
        pop it (to make number larger).
      * Append current digit.
    - After processing, take first k digits from stack.
    - Complexity: O(n)
    """
    to_remove = len(bank) - k
    stack = []

    for ch in bank:
        # Remove smaller digits if current digit is bigger
        while to_remove > 0 and stack and stack[-1] < ch:
            stack.pop()
            to_remove -= 1
        stack.append(ch)

    # Take first k digits
    return ''.join(stack[:k])

# Compute Part 2 result
K = 12
part2_total = sum(int(largest_joltage_subsequence(bank, K)) for bank in banks)

# -------------------------------
# OUTPUT RESULTS
# -------------------------------
#print("=== PART 1 ===")
#print("Max 2-digit joltage per bank:")
#for bank in banks:
#    print(f"{bank} -> {max_two_digit_joltage(bank)}")
print(f"Total Part 1: {part1_total}\n")

#print("=== PART 2 ===")
#print(f"Max {K}-digit joltage per bank:")
#for bank in banks:
#    seq = largest_joltage_subsequence(bank, K)
#    print(f"{bank} -> {seq}")
print(f"Total Part 2: {part2_total}")

execution_time = time.time() - start_time
print(f"\nExecution time: {execution_time:.6f} seconds")
