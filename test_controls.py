"""Manual check that the bridge responds to controller input.

In lockstep mode bridge.lua advances one frame per new sequence number, so
holding an action means pumping frames rather than sleeping.
"""

import time

ACTION_FILE = r"C:\Users\chipw\fzero-ai\action.txt"
STATE_FILE = r"C:\Users\chipw\fzero-ai\state.txt"

FRAMES_PER_SECOND = 60

sequence = 0


def send_action(action):
    global sequence
    sequence += 1
    with open(ACTION_FILE, "w") as file:
        file.write(f"{action} {sequence}")


def read_frame_count():
    try:
        with open(STATE_FILE, "r") as file:
            return int(file.read().split()[2])
    except (ValueError, IndexError, OSError):
        return -1


def hold(action, seconds, label):
    """Holds an action for roughly `seconds`, one frame at a time."""
    frames = int(seconds * FRAMES_PER_SECOND)

    for _ in range(frames):
        previous = read_frame_count()
        send_action(action)

        deadline = time.monotonic() + 1.0
        while read_frame_count() == previous:
            if time.monotonic() > deadline:
                print(f"  ! bridge stopped responding during '{label}'")
                return
            time.sleep(0.0005)

    print(label)


hold(3, 1.0, "accelerate")
hold(4, 1.0, "accelerate + left")
hold(5, 1.0, "accelerate + right")
hold(3, 2.0, "accelerate")
send_action(0)
