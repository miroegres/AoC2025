
# Optimized Python code for AoC with multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

def generate_invalid_ids(start, end):
    invalid_sum = 0
    min_len = len(str(start))
    max_len = len(str(end))

    for length in range(min_len, max_len + 1):
        for pattern_len in range(1, length // 2 + 1):
            if length % pattern_len == 0:
                repeats = length // pattern_len
                lower_pattern = 10 ** (pattern_len - 1)
                upper_pattern = 10 ** pattern_len - 1
                for pattern in range(lower_pattern, upper_pattern + 1):
                    num = int(str(pattern) * repeats)
                    if start <= num <= end:
                        invalid_sum += num
    return invalid_sum

if __name__ == "__main__":
    start_time = time.time()

    # Read input file
    with open('input.txt', 'r') as file1:
        input_data = file1.read().strip()

    ranges = input_data.split(',')

    total_sum = 0
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(generate_invalid_ids, *map(int, r.split('-'))) for r in ranges]
        for f in futures:
            total_sum += f.result()

    end_time = time.time()
    execution_time_ms = (end_time - start_time) * 1000  # Convert to milliseconds

    print("sum of invalid IDs:", total_sum)
    print(f"Execution time: {execution_time_ms:.3f} ms")
