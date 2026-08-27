import os
import subprocess
import time

# =====================================================================
# CONFIGURATION
# =====================================================================
COMMANDS_FILE = "temp/swipe_commands.txt"
DELAY_BETWEEN_WORDS = 0.5  # Seconds to wait for game animations to finish


def execute_adb_command(cmd_string):
    """Parses a raw command string and executes it safely via subprocess."""
    # Clean up whitespace and split into args for subprocess
    args = cmd_string.strip().split()
    if not args:
        return
    
    try:
        # Run the adb shell command
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        print(f"  ❌ Failed to execute: {' '.join(args)}")
        print(f"  Error details: {e.stderr.decode().strip()}")
    except FileNotFoundError:
        print("  ❌ System Error: 'adb' executable not found in system PATH.")


def run_macro_execution():
    """Reads swipe_commands.txt and replays them on the connected device."""
    if not os.path.exists(COMMANDS_FILE):
        raise FileNotFoundError(
            f"'{COMMANDS_FILE}' not found. Please run Step 2 (solver.py) first!"
        )

    print("Checking for connected ADB devices...")
    try:
        devices_check = subprocess.run(
            ["adb", "devices"], capture_output=True, text=True, check=True
        )
        lines = [line.strip() for line in devices_check.stdout.strip().split("\n") if line.strip()]
        
        # 'adb devices' output has a header line, so len > 1 means a device is connected
        if len(lines) <= 1:
            print("⚠️ Warning: No ADB devices detected. Please plug in your phone and enable USB debugging.")
            return
        print(f"Device ready:\n{lines[1]}")
    except Exception as e:
        print(f"Could not contact ADB: {e}")
        return

    print(f"\nParsing commands from '{COMMANDS_FILE}'...")
    with open(COMMANDS_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_word = "Unknown"
    
    print("\n🚀 Beginning swipe sequence. Keep your phone screen on!")
    for line in lines:
        line_clean = line.strip()
        
        # Skip empty lines or the bash header
        if not line_clean or line_clean.startswith("#!/bin/bash"):
            continue
            
        # Track which word we are currently executing for terminal output
        if line_clean.startswith("# Word:"):
            current_word = line_clean.replace("# Word:", "").strip()
            print(f"✍️ Swiping word: {current_word}...")
            continue
            
        # Execute the actual adb action
        if line_clean.startswith("adb"):
            execute_adb_command(line_clean)
            # Give the game UI a brief pause to register the line clear animation
            time.sleep(DELAY_BETWEEN_WORDS)

    print("\n🎉 Automation sequence finished!")


if __name__ == "__main__":
    try:
        run_macro_execution()
    except KeyboardInterrupt:
        print("\n🛑 Execution paused by user.")
    except Exception as e:
        print(f"\nExecution failed: {e}")
