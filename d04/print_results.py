
# AoC - Animate only map frames from output.txt (clear screen between frames)
import os
import time
from pathlib import Path
import re

OUTPUT_FILE = "output.txt"
FRAME_DELAY = 0.5  # seconds between frames; adjust as needed

def clear_screen():
    """Clear terminal screen and move cursor to top."""
    # Fast ANSI clear; works in Windows Terminal, PowerShell, most terminals.
    # If this doesn't work in your environment, replace with os.system('cls').
    print("\033[2J\033[H", end="")

def read_lines(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    return text.splitlines()

# Lines to drop (headers, labels, progress)
DROP_PREFIXES = (
    "Part 1 accessible:",
    "Part 1 execution time:",
    "Part 2 total removed:",
    "Part 2 execution time:",
    "=== Round",                      # e.g., "=== Round 65 ==="
    "Accessible (R):",
    "After removal (X just removed):",
    "Removed this round:",
    "Total removed so far:",
)

def is_grid_line(line: str) -> bool:
    """Grid lines contain only '.', '@', 'R', 'X'."""
    return bool(line) and re.fullmatch(r"[.@RX]+", line) is not None

def filter_non_grid(lines: list[str]) -> list[str]:
    """Remove empty lines and lines with headers/labels/progress."""
    keep = []
    for ln in lines:
        if not ln.strip():
            continue
        if any(ln.startswith(p) for p in DROP_PREFIXES):
            continue
        keep.append(ln)
    return keep

def parse_frames(lines: list[str]) -> list[list[str]]:
    """Group consecutive grid lines into frames."""
    frames = []
    cur = []
    for ln in lines:
        if is_grid_line(ln):
            cur.append(ln)
        else:
            if cur:
                frames.append(cur)
                cur = []
    if cur:
        frames.append(cur)
    return frames

def animate(frames: list[list[str]], delay: float):
    """Render frames with clear-screen between each."""
    for grid in frames:
        clear_screen()
        print("\n".join(grid))
        time.sleep(delay)

def main():
    lines = read_lines(OUTPUT_FILE)
    filtered = filter_non_grid(lines)
    frames = parse_frames(filtered)
    if not frames:
        clear_screen()
        print("No grid frames found in output.txt")
        return
    animate(frames, FRAME_DELAY)

if __name__ == "__main__":
    main()
