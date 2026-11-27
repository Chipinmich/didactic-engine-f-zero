-- Bridge between EmuHawk and the Python RL agent.
--
-- Protocol
--   action.txt  "<action> <sequence>"        written by Python
--   state.txt   "<speed> <resets> <frames>"  written by us
--   frame       PNG bytes in a memory-mapped file (see MMF_NAME)
--
-- The frame goes through shared memory rather than a PNG on disk: writing a
-- file every frame was both slow and racy, since Python could read a
-- half-written PNG and silently get a black image.

-- Training runs several emulators at once, and each needs its own endpoints.
-- BizHawk's --lua= takes no arguments, so the instance id arrives through
-- the environment instead; tools/launch_instances.py sets it per process.
-- Unset means the single-emulator default, matching instance_paths(None) in
-- bridge_client.py.
local instance = os.getenv("FZERO_INSTANCE")
local suffix = instance or ""
local model_label = os.getenv("FZERO_LABEL")

local base = "C:\\Users\\chipw\\fzero-ai\\"

local action_file = base .. "action" .. suffix .. ".txt"
local state_file = base .. "state" .. suffix .. ".txt"
local frame_file = base .. "frames\\frame" .. suffix .. ".png"

local MMF_NAME = "fzero_frame"
if instance then
    MMF_NAME = MMF_NAME .. "_" .. instance
end

-- Set false to fall back to writing PNGs to disk (slower, but easy to
-- eyeball and useful if the memory-mapped path misbehaves).
local USE_MMF = true

-- Lockstep: advance only once Python has sent a new action, so no frame is
-- ever skipped and runs are reproducible. The emulator then runs at Python's
-- pace rather than realtime, which is normal and expected for training.
local LOCKSTEP = true

-- If Python goes quiet for this many yields, resume free-running rather than
-- leaving the emulator wedged forever.
local WAIT_TIMEOUT = 600

-- Slot the reset action loads. Slot 1 is the hand-saved start line; a
-- purpose-made "centred and up to speed" state can be built into another
-- slot with tools/make_start_state.py and selected here, so every episode
-- begins from an identical, well-conditioned position.
local savestate_slot = tonumber(os.getenv("FZERO_START_SLOT") or "1")

-- Slot the save action writes to, used only when building a start state.
local save_slot = tonumber(os.getenv("FZERO_SAVE_SLOT") or "2")

-- Actions are three packed dimensions: steer * 6 + throttle * 3 + shoulder.
-- Throttle has only two values (coast, accelerate) because F-Zero has no
-- brake button -- measured, see tools/find_buttons.py. See encode_action().
local reset_action = 18

-- Verified with tools/find_buttons.py: B is the only button that
-- accelerates (+1916 from a standstill; every other button did nothing).
-- There is no brake button -- Y/A/X/L/R all shed 80 units a frame, exactly
-- the same as releasing the accelerator.
local accel_button = "B"

local reset_count = 0
local frame_count = 0

-- Declared up here because write_state() reports it. A local declared
-- AFTER a function is not visible inside it -- Lua silently reads a global
-- instead, which is nil, and the state line then carried the text "nil".
-- The Python side threw parsing that, fell back to cached values, and every
-- reset timed out.
local save_count = 0
local last_sequence = -1
local prev_action = 0

-- BizHawk sizes the mapping to the first payload it writes. A 256x224 frame
-- is usually ~12KB, but a busier frame compresses larger, overflows the
-- mapping and throws a .NET exception out of comm.mmfScreenshot() -- which
-- killed the script mid-race every time. Pre-sizing it to something no
-- frame will exceed avoids that.
local MMF_CAPACITY = 262144

local mmf_failures = 0

-- Each emulator was stuck at about 30 frames a second no matter how many
-- ran at once, while the machine sat at a fraction of its cores. That is a
-- per-instance ceiling, not a CPU limit: BizHawk throttles to realtime and
-- syncs to the sound card, so a lockstep round trip lands just past a frame
-- boundary and catches every second frame. Training never needs to hear or
-- see this, so let it run flat out.
-- SPEEDMODE CHANGES HOW INPUTS REACH THE GAME. Do not raise this for
-- throughput without re-measuring acceleration.
--
-- It was 1000. Holding accelerate from the same savestate for 120 emulated
-- frames, identical inputs every frame, the machine gains:
--
--     speedmode    25    100    200    300    400   1000
--     speed/frame  6.47  6.45   5.70   4.82   3.28   1.90
--     peak speed   1558  1556   1464   1354   1172   1106
--
-- Same emulated frames, same actions -- only how fast the host runs them.
-- Above 100 the core stops applying every input, so training was driving on
-- roughly a third of the throttle it asked for.
--
-- Acceleration alone originally made 25 and 100 look equivalent, but a direct
-- fixed-frame steering replay exposed the missing test: at 16 turn frames,
-- 50% produced a 17.05 +/- 2.16 world-unit left/right span, while 100%
-- produced 15.19 +/- 5.60. Thus 100% was about 11% weaker and 2.6x less
-- consistent. 25% and 50% steering matched, so 50% is the fastest measured
-- full-fidelity mode.
--
-- The throughput this cost is not real. Single-instance it is 59.5 frames a
-- second here against 90.2 at 1000, but eight instances are limited by the
-- single Python process driving them, which measures ~36 frames a second each
-- -- comfortably under the 59.5 cap. See the note above about sound: turning
-- the sound card sync off is what lifted a per-instance ceiling of ~30, and
-- that fix still stands on its own.
local speed_mode = tonumber(os.getenv("FZERO_SPEEDMODE") or "50")

pcall(function() client.SetSoundOn(false) end)
pcall(function() client.speedmode(speed_mode) end)

if USE_MMF then
    comm.mmfSetFilename(MMF_NAME)
    pcall(function()
        comm.mmfWrite(MMF_NAME, string.rep("\0", MMF_CAPACITY))
    end)
end


-- Optional work-RAM dump, for hunting addresses (tools/find_heading.py).
-- Off unless FZERO_RAM_DUMP is set, because it writes 128KB a sample.
local ram_dump_path = os.getenv("FZERO_RAM_DUMP")
local ram_dump_every = tonumber(os.getenv("FZERO_RAM_EVERY") or "2")
local ram_file = nil
local wram_size = 0x20000

if ram_dump_path then
    ram_file = io.open(ram_dump_path, "wb")
end


local function dump_ram()
    if not ram_file or frame_count % ram_dump_every ~= 0 then
        return
    end

    local block = memory.readbyterange(0, wram_size, "WRAM")

    -- Older BizHawk builds hand back a table rather than a string.
    if type(block) == "table" then
        local bytes = {}
        for i = 0, wram_size - 1 do
            bytes[i + 1] = string.char(block[i] or 0)
        end
        block = table.concat(bytes)
    end

    ram_file:write(block)
end


-- Optional per-frame log of the action actually applied to the pad, for
-- checking that a held input really is held rather than flickering. Off
-- unless FZERO_ACTION_LOG is set.
local action_log_path = os.getenv("FZERO_ACTION_LOG")
local action_log = nil
if action_log_path then
    action_log = io.open(action_log_path, "w")
    if action_log then
        action_log:write("frame,action,steer,throttle,shoulder,waited\n")
    end
end


local function capture_frame()
    if not USE_MMF then
        client.screenshot(frame_file)
        return
    end

    -- Never let a capture failure kill the run: fall back to disk and let
    -- Python see the failure count so it can read frames from there.
    if not pcall(comm.mmfScreenshot) then
        mmf_failures = mmf_failures + 1
        client.screenshot(frame_file)
    end
end


-- Visible ownership label for multi-emulator training. It is deliberately
-- above every vision scan row (the first is y=75), so displaying it cannot
-- alter road edges or seven-ray observations even when screenshots include
-- BizHawk's GUI layer.
local function draw_model_label()
    if not model_label then
        return
    end
    local text = model_label
    if instance then
        text = text .. " | INSTANCE " .. instance
    end
    pcall(function()
        gui.text(4, 4, text, "white", "black")
    end)
end

-- Drawing only once immediately before frameadvance can flicker because the
-- lockstep bridge yields many times while waiting for Python, and BizHawk may
-- clear its GUI layer between those yields. A frame-end callback redraws the
-- label on every displayed frame and leaves the most recent frame labelled
-- while the emulator is waiting.
local persistent_label_callback = false
if model_label then
    persistent_label_callback = pcall(function()
        event.onframeend(draw_model_label, "fzero_model_instance_label")
    end)
end


local function read_action()
    local file = io.open(action_file, "r")
    if not file then
        return 0, last_sequence
    end

    local contents = file:read("*all")
    file:close()

    local action, sequence = contents:match("^%s*(%-?%d+)%s+(%-?%d+)")
    if action then
        return tonumber(action), tonumber(sequence)
    end

    -- Tolerate a bare action with no sequence number, so the manual test
    -- scripts still work against this bridge.
    return tonumber(contents) or 0, last_sequence
end


-- The lap counter. 0x01161 mirrors it and is the cross-check.
--
-- NOT 0x000C0, which was used before: 0x000C0/0x000C1 are a two-field packed
-- BCD CLOCK, 0x000C1 running 0..89 and carrying into 0x000C0. On a recorded
-- human drive that carry fell at frame 5135, twenty frames from a real lap
-- boundary at 5155 -- close enough to pass for a lap counter.
--
-- 0x00F53 ticks at frames 3255, 5155, 6800 and 8515 of that recording, and the
-- machine returns to its lap-start position at 3235, 5140, 6785 and 8500:
-- agreement within ~20 frames on all four laps.
local lap_address = 0x00F53

-- 0x000C1 IS THE LAP TIMER, NOT PROGRESS. Do not use it for position.
--
-- It was originally accepted because it "climbs to exactly 89 and resets to 0
-- at every lap change, with no decrease anywhere". Every word of that is true
-- -- and equally true of a clock, because a lap timer and a lap position have
-- the same sawtooth shape. Shape ranking alone cannot separate them.
--
-- The test that does: hold the car still in the middle of a race. Measured
-- with the throttle released, speed read exactly 0 for 124 consecutive frames
-- and the integral of speed never moved, while this byte counted 18 -> 26. It
-- also ticks on a fixed ~15-frame cadence regardless of speed, and it is
-- packed BCD, which is why raw values jump 25 -> 32 and 40 -> 49.
--
-- Roughly 40,000 training episodes were scored on this as "distance". Fitness
-- was rewarding elapsed time, which is why every genetic run went flat and why
-- laps were occasionally "completed" by policies that never drove anywhere.
-- Progress now comes from the machine's world position instead; see
-- position_x_address below and track_map.py.
local progress_address = 0x000C1

-- Whether the player's race is actually live. Reads 2 while racing and 1 on
-- every menu, results and title screen.
--
-- This exists because the picture cannot answer the question. The
-- "1. MUTE CITY I" course card is the same flat road-grey as the track, so
-- the road-pixel test reads it as a wide open road; the "YOU LOST" screen
-- measures right in the middle of the racing distribution on every pixel
-- statistic tried. The agent was steering against menus and training on it.
--
-- Found with tools/find_race_state.py and confirmed on data the search
-- never saw: across a 2000-frame boot capture -- title, menus, and the
-- attract-mode demo racing at speed 2104 -- this reads 1 on 100% of frames.
-- A demo race at full speed is exactly the case that defeats every
-- movement-based test, so that is the one that matters.
--
-- 0x00C15, 0x00C17, 0x00C19 and 0x010AF agree on all 5035 captured samples
-- and are the cross-check if this ever looks wrong. 0x00093 looks like the
-- same flag but is not: it reads "racing" on 48% of the boot capture.
local racing_address = 0x00DC8
local RACING_VALUE = 2

-- The machine's world position. Progress round the lap is derived from these
-- on the Python side (see track_map.py), because F-Zero stores no
-- lap-position counter: a search over all 131072 work-RAM addresses for a
-- per-lap sawtooth that also freezes when the car freezes returned only the
-- lap timer.
--
-- These two are a genuine orthogonal pair, which matters: several nearby
-- addresses track position but are redundant with each other (correlation
-- 0.999), and a pair like that traces a line out and back rather than a loop,
-- folding the track onto itself.
--
-- The test that picks the real pair is physical: for true coordinates the
-- per-frame displacement magnitude must be proportional to speed. Over a
-- recorded human lap this pair scores corr(|dp|, speed) = +0.9922 with
-- corr(x, y) = +0.385, while every other candidate pair scores 0.28 with
-- |corr(x, y)| = 0.999.
--
-- Resampled into 90 checkpoints the lap closes on itself to 0.72%, and the
-- closest the centreline comes to itself is 504 units against a 168-unit
-- checkpoint spacing -- so a position maps to one checkpoint unambiguously.
local position_x_address = 0x00B70
local position_y_address = 0x00B90

local function write_state()
    local speed = memory.read_u16_le(0x000B20)
    local lap = memory.readbyte(lap_address)
    -- Still published, but it is the LAP TIMER, not progress -- kept only as
    -- a diagnostic. See progress_address's comment.
    local lap_time = memory.readbyte(progress_address)
    local racing = (memory.readbyte(racing_address) == RACING_VALUE) and 1 or 0
    local position_x = memory.read_u16_le(position_x_address)
    local position_y = memory.read_u16_le(position_y_address)

    local output = io.open(state_file, "w")
    if output then
        output:write(
            tostring(speed) .. " " ..
            tostring(reset_count) .. " " ..
            tostring(frame_count) .. " " ..
            tostring(lap) .. " " ..
            tostring(lap_time) .. " " ..
            tostring(mmf_failures) .. " " ..
            tostring(save_count) .. " " ..
            tostring(racing) .. " " ..
            tostring(position_x) .. " " ..
            tostring(position_y) .. " " ..
            -- Published so the Python side can refuse to drive a policy under
            -- a different host speed than it was trained under. Above 50 the
            -- emulator stops being deterministic (see speed_mode above), so a
            -- silent mismatch would change the physics out from under a
            -- trained model without anything reporting it.
            tostring(speed_mode)
        )
        output:close()
    end
end


-- Outside the action space: press one named button alone, so
-- tools/find_buttons.py can identify what each button does.
local raw_button_base = 19
local raw_buttons = {"B", "Y", "A", "X", "L", "R"}

-- Writes the current state to save_slot. Outside the action space; used
-- once when building a start state, never during training.
local save_action = 25
local load_slot_base = 32
local save_slot_base = 48


local function apply_action(action)
    local buttons = {}

    if action >= raw_button_base then
        local name = raw_buttons[action - raw_button_base + 1]
        if name then
            buttons[name] = true
        end
        joypad.set(buttons, 1)
        return
    end

    local steer = math.floor(action / 6) % 3
    local throttle = math.floor(action / 3) % 2
    local shoulder = action % 3

    if steer == 1 then
        buttons["Left"] = true
    elseif steer == 2 then
        buttons["Right"] = true
    end

    if throttle == 1 then
        buttons[accel_button] = true
    end

    if shoulder == 1 then
        buttons["L"] = true
    elseif shoulder == 2 then
        buttons["R"] = true
    end

    joypad.set(buttons, 1)
end


-- The script has been dying mid-run with no visible cause, so any error is
-- written where the Python side can read it rather than only appearing in
-- the Lua Console.
local error_file = base .. "bridge_error" .. suffix .. ".txt"

os.remove(error_file)

local function main_loop()

while true do

    local action, sequence = read_action()

    -- =========================
    -- WAIT FOR PYTHON
    -- =========================

    local last_wait = 0
    if LOCKSTEP then
        local waited = 0

        while sequence == last_sequence and waited < WAIT_TIMEOUT do
            emu.yield()
            waited = waited + 1
            action, sequence = read_action()
        end
        last_wait = waited
    end

    last_sequence = sequence


    -- =========================
    -- RESET OR ACT
    -- =========================

    if action >= save_slot_base and action < save_slot_base + 10 then

        if prev_action ~= action then
            savestate.saveslot(action - save_slot_base)
            save_count = save_count + 1
        end

        joypad.set({}, 1)

    elseif action >= load_slot_base and action < load_slot_base + 10 then

        if prev_action ~= action then
            savestate.loadslot(action - load_slot_base)
            reset_count = reset_count + 1
        end

        joypad.set({}, 1)

    elseif action == save_action then

        -- Edge-triggered like the reset, so holding it does not write the
        -- slot on every frame.
        if prev_action ~= save_action then
            savestate.saveslot(save_slot)
            save_count = save_count + 1
        end

        joypad.set({}, 1)

    elseif action == reset_action then

        -- Edge-triggered: Python holds the reset action until it sees
        -- reset_count change, so without this we would reload every frame.
        if prev_action ~= reset_action then
            savestate.loadslot(savestate_slot)
            reset_count = reset_count + 1
        end

        joypad.set({}, 1)
    else
        apply_action(action)
    end

    if action_log then
        action_log:write(
            frame_count .. "," .. action .. "," ..
            (math.floor(action / 6) % 3) .. "," ..
            (math.floor(action / 3) % 2) .. "," ..
            (action % 3) .. "," .. tostring(last_wait) .. "\n"
        )
        if frame_count % 40 == 0 then action_log:flush() end
    end

    prev_action = action


    -- =========================
    -- ADVANCE, THEN CAPTURE
    -- =========================

    -- Compatibility fallback for BizHawk builds without event.onframeend.
    if not persistent_label_callback then
        draw_model_label()
    end
    emu.frameadvance()
    frame_count = frame_count + 1

    capture_frame()
    dump_ram()

    write_state()
end

end  -- main_loop


write_state()

local ok, err = pcall(main_loop)

if not ok then
    local message = tostring(err)
    console.log("bridge.lua ERROR: " .. message)

    local log = io.open(error_file, "w")
    if log then
        log:write(message)
        log:close()
    end
end
