# python code fo AoC
import re
import time

start_time = time.time()

# Read input file
with open('input.txt', 'r') as file1:
    input_data = file1.read().strip()  # Read entire file and remove extra spaces/newlines

#input_data = "11-22,95-115,998-1012,1188511880-1188511890,222220-222224," \
#             "1698522-1698528,446443-446449,38593856-38593862,565653-565659," \
#             "824824821-824824827,2121212118-2121212124"

ranges = input_data.split(',')

total_sum = 0

def is_invalid_id(num):
    s = str(num)
    if len(s) % 2 == 0:  # Only even length can be repeated pattern
        mid = len(s) // 2
        if s[:mid] == s[mid:]:
            return True
    return False

for r in ranges:
    start, end = map(int, r.split('-'))
    for num in range(start, end + 1):
        if is_invalid_id(num):
            total_sum += num


end_time = time.time()
execution_time = end_time - start_time


print("sum of invalid IDs:", total_sum)
print(f"Execution time: {execution_time:.6f} seconds")