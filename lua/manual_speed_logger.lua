-- Passive speed recorder for a human-driven F-Zero run.
-- It never calls joypad.set(), changes emulator speed, or advances extra
-- frames, so keyboard/controller input remains entirely with the player.

local output_path = "C:/Users/chipw/fzero-ai/runs/manual_speed.csv"
local output = assert(io.open(output_path, "w"))
output:write("frame,speed,lap,progress,racing,B,A,L,R,Left,Right\n")

local frame = 0
local peak = 0

local function pressed(buttons, name)
    return (buttons[name] and 1) or 0
end

local function close_output()
    if output then
        output:flush()
        output:close()
        output = nil
    end
end

event.onexit(close_output, "close manual speed recording")

while true do
    local buttons = joypad.get(1)
    local speed = memory.read_u16_le(0x000B20)
    local lap = memory.readbyte(0x000C0)
    local progress = memory.readbyte(0x000C1)
    local racing = memory.readbyte(0x000DC8)
    peak = math.max(peak, speed)

    output:write(string.format(
        "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
        frame, speed, lap, progress, racing,
        pressed(buttons, "B"), pressed(buttons, "A"),
        pressed(buttons, "L"), pressed(buttons, "R"),
        pressed(buttons, "Left"), pressed(buttons, "Right")
    ))
    if frame % 60 == 0 then
        output:flush()
    end

    gui.text(2, 2, "REC speed " .. tostring(speed) .. "  peak " .. tostring(peak))
    frame = frame + 1
    emu.frameadvance()
end
