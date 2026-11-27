-- Records SNES work RAM while YOU drive, so tools/find_ram.py can work out
-- which addresses hold lap count, progress round the track, and power.
--
-- This is deliberately NOT the lockstep bridge: it free-runs and never
-- touches the controller, so you drive normally with the keyboard.
--
-- Load this INSTEAD of bridge.lua (Lua Console -> Script -> Open Script),
-- drive two or three complete laps, then stop the script and run:
--     python tools/find_ram.py

local out_path = "C:\\Users\\chipw\\fzero-ai\\ram_capture.bin"
local meta_path = "C:\\Users\\chipw\\fzero-ai\\ram_capture.csv"

-- SNES work RAM is 128KB. Dumping every frame would be far more data than
-- the analysis needs, so sample a few times a second.
local wram_size = 0x20000
-- 5 rather than 15: the lap rollover has to be caught precisely enough
-- to read the position byte's maximum before it wraps.
local sample_every = 5

local speed_address = 0x000B20

-- Candidate track-position bytes, logged straight into the CSV so the lap
-- structure is readable without touching the 128KB-a-sample binary.
--
-- 0x000C1 is the address bridge.lua currently calls "progress". It is a
-- packed-BCD clock: measured on a parked car it kept counting 18 -> 26 over
-- 124 frames while the speed reading sat at exactly 0 and the distance
-- integral never moved. The other four freeze when the car freezes.
local candidate_addresses = {0x00F53, 0x000C0, 0x000C1}

local out = io.open(out_path, "wb")
local meta = io.open(meta_path, "w")

if not out or not meta then
    console.log("ram_record: could not open output files")
    return
end

meta:write("sample,frame,speed,a3,a9,b71,c0,c1\n")

console.log("ram_record: capturing. Drive 2-3 full laps, then stop the script.")
console.log("  binary -> " .. out_path)
console.log("  index  -> " .. meta_path)

local frame = 0
local sample = 0

while true do

    if frame % sample_every == 0 then
        -- memory.readbyterange returns the block far faster than reading
        -- each address individually, which matters at 128KB a sample.
        local block = memory.readbyterange(0, wram_size, "WRAM")

        -- Older BizHawk builds hand back a table rather than a string.
        if type(block) == "table" then
            local bytes = {}
            for i = 0, wram_size - 1 do
                bytes[i + 1] = string.char(block[i] or 0)
            end
            block = table.concat(bytes)
        end

        out:write(block)

        local speed = memory.read_u16_le(speed_address)
        local row = sample .. "," .. frame .. "," .. speed
        for _, address in ipairs(candidate_addresses) do
            row = row .. "," .. memory.readbyte(address)
        end
        meta:write(row .. "\n")

        sample = sample + 1

        if sample % 40 == 0 then
            out:flush()
            meta:flush()
            console.log("ram_record: " .. sample .. " samples")
        end
    end

    frame = frame + 1
    emu.frameadvance()
end
