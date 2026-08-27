import os, sys
import time
import cv2
from image_processor import capture_screen, extract_game_data, export_extracted_data
from solver import solve_level_with_brute_force, generate_adb_swipe_commands, load_game_state_sequence
from swiper import execute_adb_command

import multiprocessing
from check_action import detect_logo, next_action, screenshot_listener

SWIPE_DELAY = 1.5


def play_level():
	"""Runs a single iteration of capturing, solving, and swiping."""
	print("\n" + "="*50)
	print("🎬 STEP 1: Capturing screen and parsing layout...")
	print("="*50)
	
	# 1. Capture screen from the device
	if not capture_screen():
		print("❌ Screenshot failed. Is your device connected via ADB?")
		return False

	# 2. Extract OCR and template matching sequence data
	try:
		words, grid, coord_grid, rows, cols, mode = extract_game_data()
	except Exception as e:
		print(f"❌ Error extracting game data: {e}")
		return False

	# Export raw data for debug safety
	export_extracted_data(words, grid, coord_grid)

	print("\n" + "="*50)
	print("🧩 STEP 2: Alphabetical Brute-Force Tracing...")
	print("="*50)

	# 3. Hand off the layout sequence, grid, and dimensions to the brute-force solver
	# solved_paths = solve_level_with_brute_force(words, grid, rows, cols) # wrong command by Gemini
	sequence, grid, coords = load_game_state_sequence()
	ROW_MAX = len(grid)
	COL_MAX = len(grid[0]) if ROW_MAX > 0 else 0
		
	solved_paths = solve_level_with_brute_force(sequence, grid, ROW_MAX, COL_MAX, mode)
	
	# 4. Generate the direct swipe command strings
	swipe_commands = generate_adb_swipe_commands(solved_paths, coord_grid)
	
	if not swipe_commands:
		print("⚠️ No words were successfully matched to paths on the grid.")
		return True

	print("\n" + "="*50)
	print("⚡ STEP 3: Executing sweeps automatically...")
	print("="*50)

	# 5. Play each swipe on the phone
	for command_block in swipe_commands:
		for line in command_block.strip().split("\n"):
			line_clean = line.strip()
			
			if not line_clean:
				continue
				
			if line_clean.startswith("# Word:"):
				word_label = line_clean.replace("# Word:", "").strip()
				print(f"👉 Swiping word: {word_label}")
				continue
				
			if line_clean.startswith("adb"):
				execute_adb_command(line_clean)
				time.sleep(SWIPE_DELAY)
	
		#in case the level is finished but the program still swipe the answers
		if detect_logo("temp/test-home.png", "templates/buttons/home.png") or detect_logo("temp/test-next-level.png", "templates/buttons/next-level.png"):
			break

	print("\n🎉 Level completed!")
	return True


def main():
	print("==================================================")
	print("	  ANDROID WORD SEARCH AUTOMATIC SOLVER		")
	print("==================================================")
	print("Make sure USB debugging is active and your game is ")
	print("open on the grid screen.\n")
	
	while True:
		p_next_action = multiprocessing.Process(target=next_action, daemon=True) # turn on the detect start/continue button at all time
		p_screenshot_listener = multiprocessing.Process(target=screenshot_listener, daemon=True) # turn on the detect start/continue button at all time
		p_next_action.start()		
		p_screenshot_listener.start()					
		try:
			#os.system('cls' if os.name == 'nt' else 'clear') ### uncomment when complete
			user_input = input("Ready! Press [Enter] to start the level (or type 'q' and press [Enter] to quit): ")
			

			if user_input.strip().lower() == 'q':
				print("\n👋 Exiting wrapper script.")
				p_next_action.kill()
				p_screenshot_listener.kill()
				p_next_action.join()	
				p_screenshot_listener.join()
				break
				
			# Execute the automation pipeline
			success = play_level()
			
			if not success:
				print("\n⚠️ An error occurred during automated play.")
				
			print("\n" + "~"*50)
			p_next_action.kill()
			p_screenshot_listener.kill()
			p_next_action.join()
			p_screenshot_listener.join()			
			
		except KeyboardInterrupt:
			print("\n\n🛑 Automation stopped by user (Ctrl+C). Bye!")
			sys.exit(0)

if __name__ == "__main__":
	main()
