
# python code for AoC
import time

start_time = time.time()

# Read input file
with open('input.txt', 'r') as file1:
    input_data = file1.read().strip()  # Read entire file and remove extra spaces/newlines

ranges = input_data.split(',')

total_sum = 0

def is_invalid_id(num):
    s = str(num)
    # Check if s is made of a repeating pattern at least twice
    return s in (s + s)[1:-1]

for r in ranges:
    start, end = map(int, r.split('-'))
    for num in range(start, end + 1):
        if is_invalid_id(num):
            total_sum += num

end_time = time.time()
execution_time = end_time - start_time

print("sum of invalid IDs:", total_sum)
print(f"Execution time: {execution_time:.6f} seconds")
