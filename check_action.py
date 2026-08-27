from snapshot import capture_direct_android_snapshot
import subprocess
import pynput
from pynput.keyboard import Key, Controller
import time, sys
import os
import cv2

def detect_logo(image_file, button_path, confidence_threshold = 0.9):
	# This function check if the next level button is ready
	
	# STEP 1: Load your template library
	template_img = cv2.imread(button_path, cv2.IMREAD_GRAYSCALE)	

	# Read snapshot in grayscale for template verification
	img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
		
	result = cv2.matchTemplate(img, template_img, cv2.TM_CCOEFF_NORMED)
	_, max_val, _, _ = cv2.minMaxLoc(result)
	
	highest_score = -1	
				
	if max_val > highest_score:
		highest_score = max_val
				
	# STEP 2: see if the two image matches
	if highest_score >= confidence_threshold:
		return True			

	else:
		return False

ADB_PATH = "adb" 

def send_tap(x, y):
	"""Sends an ADB tap command to the connected Android device."""
	cmd = [ADB_PATH, "shell", "input", "tap", str(x), str(y)]
	try:
		subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
		# print(f"Executed Tap: ({x}, {y})")

	except subprocess.CalledProcessError:
		print("Error: Failed to execute ADB command. Is your device connected?")

def next_action():			
	# click the play button if it is there	
	while True:
		try:
			keyboard = Controller()			

			capture_direct_android_snapshot("temp/test-awesome.png", 375, 1900, 330, 70)
			capture_direct_android_snapshot("temp/test-close.png", 977, 354, 94, 94)
			capture_direct_android_snapshot("temp/test-collect.png", 385, 1665, 320, 95)
			capture_direct_android_snapshot("temp/test-continue.png", 390, 1700, 300, 90)
			capture_direct_android_snapshot("temp/test-home.png", 455, 2115, 170, 200)
			capture_direct_android_snapshot("temp/test-next-level.png", 380, 1670, 320, 80) 
			time.sleep(1) # waiting for some random pop-up
			
			if detect_logo("temp/test-awesome.png", "templates/buttons/awesome.png"):
				send_tap(540,1950) # badge awesome button
				continue

			elif detect_logo("temp/test-close.png", "templates/buttons/close.png"):
				send_tap(1000,400) # close the award screen
				continue

			elif detect_logo("temp/test-collect.png", "templates/buttons/collect.png"):
				send_tap(540,1700) # collect stamp button
				continue

			elif detect_logo("temp/test-continue.png", "templates/buttons/continue.png"):
				send_tap(540,1750) # continue to next level button
				continue
	
			elif detect_logo("temp/test-home.png", "templates/buttons/home.png"):
				send_tap(540,1900) # play button from home
				capture_direct_android_snapshot("temp/test-home.png", 455, 2115, 170, 200)
				time.sleep(5)
				keyboard.press(Key.enter)
				keyboard.release(Key.enter)
				continue

			elif detect_logo("temp/test-next-level.png", "templates/buttons/next-level.png"):
				send_tap(540,1700) # play button from home
				capture_direct_android_snapshot("temp/test-next-level.png", 380, 1670, 320, 80) 
				time.sleep(5)
				keyboard.press(Key.enter)
				keyboard.release(Key.enter)	
				continue 	
			
		except KeyboardInterrupt:
			sys.exit(0)

def screenshot_listener():
	def on_press(key):	
		# handle screen shot by pressing spacebar
		if key == pynput.keyboard.Key.space:
			capture_direct_android_snapshot("temp/current-level.png", 0, 0, 1080, 2340)
	try:
		with pynput.keyboard.Listener(on_press=on_press) as listener:
			listener.join()

	except KeyboardInterrupt:
			sys.exit(0)



