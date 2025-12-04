
# AoC - Animate output.txt rounds in terminal
import os
import time
from pathlib import Path

OUTPUT_FILE = "output.txt"
FRAME_DELAY = 0.6  # seconds between frames; tweak to taste

def clear_screen():
    # Cross-platform screen clear
    if os.name == "nt":  # Windows
        os.system("cls")
    else:  # macOS/Linux/Unix
        os.system("clear")

def read_output_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")

def parse_frames(log_text: str):
    """
    Parse output.txt into a list of frames.
    Each frame is a tuple: (title, body_text)
    We try to capture:
      - Header lines (timings/summary) as first frame
      - For each round:
          Frame A: 'Accessible (R):' grid
          Frame B: 'After removal (X just removed):' grid + round stats
    """
    lines = log_text.splitlines()
    frames = []

    # Optional header: everything before the first "=== Round"
    header_lines = []
    i = 0
    while i < len(lines) and not lines[i].startswith("=== Round"):
        header_lines.append(lines[i])
        i += 1
    if header_lines:
        frames.append(("Summary", "\n".join(header_lines)))

    # Parse rounds
    while i < len(lines):
        # Expect "=== Round N ==="
        if lines[i].startswith("=== Round"):
            round_title = lines[i].strip()
            i += 1

            # Expect "Accessible (R):" followed by a grid block until blank line
            if i < len(lines) and lines[i].startswith("Accessible (R):"):
                i += 1  # skip the label
                grid_R = []
                while i < len(lines) and lines[i].strip() != "":
                    grid_R.append(lines[i])
                    i += 1
                # Skip the blank line
                if i < len(lines) and lines[i].strip() == "":
                    i += 1
                frames.append((f"{round_title} — Accessible (R)", "\n".join(grid_R)))

            # Expect "After removal (X just removed):" + grid until blank
            # then two stat lines and a blank
            grid_X = []
            if i < len(lines) and lines[i].startswith("After removal (X just removed):"):
                i += 1  # skip the label
                while i < len(lines) and lines[i].strip() != "":
                    grid_X.append(lines[i])
                    i += 1
                # Skip the blank line
                if i < len(lines) and lines[i].strip() == "":
                    i += 1

                # Grab stats if present
                stats = []
                if i < len(lines) and lines[i].startswith("Removed this round:"):
                    stats.append(lines[i])
                    i += 1
                if i < len(lines) and lines[i].startswith("Total removed so far:"):
                    stats.append(lines[i])
                    i += 1
                # Skip trailing blank
                if i < len(lines) and lines[i].strip() == "":
                    i += 1

                # Combine grid + stats for the X frame
                body_X = "\n".join(grid_X + ([""] if grid_X and stats else []) + stats)
                frames.append((f"{round_title} — After removal", body_X))
        else:
            # If formatting deviates, advance to avoid infinite loops
            i += 1

    return frames

def animate_frames(frames, delay=FRAME_DELAY):
    for title, body in frames:
        clear_screen()
        print(title)
        print("-" * len(title))
        print(body)
        time.sleep(delay)

def main():
    # Read output.txt
    log_text = read_output_file(OUTPUT_FILE)

    # Parse into frames
    frames = parse_frames(log_text)

    # If no frames parsed, just print the file once
    if not frames:
        clear_screen()
        print(log_text)
        return

    # Animate!
    animate_frames(frames, delay=FRAME_DELAY)

    # Final screen: show last frame persistently for a moment
    #time.sleep(1.0)

if __name__ == "__main__":
    main()
