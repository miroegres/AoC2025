
# python code for AoC
import time

start_time = time.time()

# Read input file
with open('input.txt', 'r') as file1:
    input_data = file1.read().strip()  # Read entire file and remove extra spaces/newlines

# For now, use sample data
#input_data = """987654321111111
#811111111111119
#234234234234278
#818181911112111"""

# Split into banks
banks = input_data.splitlines()

def max_joltage_for_bank(bank: str) -> int:
    """
    Find the largest possible two-digit number formed by two digits in the bank,
    preserving their original order (cannot rearrange).
    """
    max_joltage = 0
    for i in range(len(bank)):
        for j in range(i + 1, len(bank)):
            joltage = int(bank[i] + bank[j])
            if joltage > max_joltage:
                max_joltage = joltage
    return max_joltage

# Compute total result
result = sum(max_joltage_for_bank(bank) for bank in banks)

# Print details
#print("Maximum joltage per bank:")
#for bank in banks:
#    print(f"{bank} -> {max_joltage_for_bank(bank)}")

print("result:", result)

execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.3f} seconds")
