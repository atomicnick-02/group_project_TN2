import time
import sys
import pyCandle
import numpy as np

# ==========================================
# 1. HARDWARE SETUP
# ==========================================
print("[1/3] Initializing Hardware...")
candle = pyCandle.Candle(pyCandle.CAN_BAUD_1M, True, pyCandle.USB)
ids = candle.ping(pyCandle.CAN_BAUD_1M)

if len(ids) < 2:
    sys.exit(f"Error: Needed 2 motors, found {len(ids)}.")

for m_id in ids:
    candle.addMd80(m_id)
    candle.controlMd80SetEncoderZero(m_id)
    candle.controlMd80Mode(m_id, pyCandle.RAW_TORQUE)
    candle.controlMd80Enable(m_id, True)

candle.begin()
time.sleep(1)

motor_0 = candle.md80s[1]
motor_1 = candle.md80s[0]


# ==========================================
# 4. CONTROL LOOP
# ==========================================
dt = 0.05
max_torque = 0.07          # Physical hard stop
current_idx = 0

DEVIATION_THRESHOLD = 1.2    # threshold for switching to recovery
ADVANCE_THRESHOLD = 0.4     # if close enough, advance along trajectory

# Local search window to prevent teleporting
WINDOW_FWD = 6
WINDOW_BACK = 2

# Distance metric weighting (reduces velocity dominance)
VEL_W = 0.1

try:
    while True:
        loop_start = time.time()

        # --- A. SENSE ---
        p0 = motor_0.getPosition()
        p1 = motor_1.getPosition()
        v0 = motor_0.getVelocity()
        v1 = motor_1.getVelocity()
        x_real = np.array([p0, p1, v0, v1], dtype=float)

    
        torque = np.random.uniform(low = -0.05, high = 0.05)
        motor_0.setTargetTorque(torque)
        motor_1.setTargetTorque(torque*0.5)

        elapsed = time.time() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)
        
except KeyboardInterrupt:
    print("Emergency Stop!")

finally:
    # Always disable motors cleanly
    candle.controlMd80Enable(motor_0.getId(), False)
    candle.controlMd80Enable(motor_1.getId(), False)
    candle.end()
