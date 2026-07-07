"""A tiny REPL that stands in for a harness CLI inside a tmux pane.

Echoes each received line back with ANSI color codes (which the fallback
adapter must strip) after a short delay, so output-stability idle detection
has something realistic to watch.
"""

import sys
import time


def main():
    print("FAKE HARNESS READY")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        time.sleep(0.3)
        print(f"\x1b[32mGOT:\x1b[0m {line}")
        print("\x1b[1mdone.\x1b[0m")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
