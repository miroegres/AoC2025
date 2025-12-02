
import time

start_time = time.time()

# Read input file
with open('input.txt', 'r') as file1:
    input_data = file1.read().strip()

ranges = input_data.split(',')

total_sum = 0

def generate_invalid_ids(start, end):
    invalid_sum = 0
    # Determine digit lengths in this range
    min_len = len(str(start))
    max_len = len(str(end))
    
    for length in range(min_len, max_len + 1):
        # For each possible pattern length
        for pattern_len in range(1, length // 2 + 1):
            if length % pattern_len == 0:  # Must divide evenly
                repeats = length // pattern_len
                # Generate all possible patterns for this length
                lower_pattern = 10 ** (pattern_len - 1)
                upper_pattern = 10 ** pattern_len - 1
                for pattern in range(lower_pattern, upper_pattern + 1):
                    num = int(str(pattern) * repeats)
                    if start <= num <= end:
                        invalid_sum += num
    return invalid_sum

for r in ranges:
    start, end = map(int, r.split('-'))
    total_sum += generate_invalid_ids(start, end)

end_time = time.time()
execution_time = end_time - start_time

print("sum of invalid IDs:", total_sum)
print(f"Execution time: {execution_time:.3f} ms")
