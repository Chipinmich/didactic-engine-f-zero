# F-Zero AI

An experiment in training agents to drive Mute City in the SNES version of F-Zero. The goal is to complete laps consistently. The project uses BizHawk to run the game, a Lua bridge to send controller inputs and read game state, and Python to extract road features and train the agents.

## Driving approaches

- **DQN** learns to select steering and lean actions from a discrete action set.
- **PPO** learns a driving policy through reinforcement learning.
- **Neuroevolution** evaluates and evolves driving policies across generations.
- **Autopilot** uses hand-written road-following rules as a baseline.

## Setup on Windows

1. Install Python 3.13 and BizHawk. Supply your own F-Zero (USA) ROM.
2. From the project folder, create the Python environment and install dependencies:

   ```powershell
   py -3.13 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Check the project, BizHawk, and ROM paths in the batch launcher you plan to use. Some launchers contain machine-specific paths.
4. For manual setup, load the ROM in BizHawk, open **Tools → Lua Console**, and run `lua/bridge.lua`. For a basic training start, create a savestate in slot 1 at the start of the race. Multi-start training modes may require additional slots.
5. Verify that Python can communicate with the emulator:

   ```powershell
   .\.venv\Scripts\python.exe tools\check_bridge.py
   ```

## Train and monitor

From the project folder:

```powershell
.\train_rl.bat dqn 8 500000
.\evolve.bat 30 8
```

The first command starts DQN training with eight emulator instances and a 500,000-timestep target. The second runs neuroevolution for 30 generations with eight emulator instances.

Start the dashboard separately:

```powershell
.\.venv\Scripts\python.exe tools\dashboard.py
```

Open [the local dashboard](http://localhost:8321) and select a recorded run. The **Recent Improvement Assessment** compares the latest X completed episodes with the preceding X. The window defaults to 100 and can be changed on the page. Use completed laps and race progress to judge improvement, with wall contact, steering quality, and reward to help explain the trend.

The dashboard reads local run records from `runs/store/`. Those records, model checkpoints, emulator savestates, and ROMs stay local. The Mute City track map is included in the repository.
