import time

STATE_FILE = r"C:\Users\chipw\fzero-ai\state.txt"

while True:

    try:
        with open(STATE_FILE, "r") as file:
            speed, resets, frames = file.read().split()

        print("Speed:", speed, "Resets:", resets, "Frames:", frames)

    except:
        pass

    time.sleep(0.1)
