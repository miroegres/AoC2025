# python code for AoC
import time

# Read input file
with open('input.txt', 'r') as file1:
    input_data = file1.read().strip()  # Read entire file and remove extra spaces/newlines

start_time = time.time()

# code here

execution_time = time.time() - start_time
print("result:", result)
print(f"Execution time: {execution_time:.3f} seconds")
