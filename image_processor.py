import os
import subprocess
import string
import cv2
import easyocr
import numpy as np
import re

# =====================================================================
# CONFIGURATION & CONSTANTS (Target Resolution: 1080 x 2340)
# =====================================================================
DEFAULT_IMAGE_NAME = "temp/game.png"
TEMPLATE_DIR = "templates/letters"
EXPORT_BANK_FILE = "temp/bank.txt"
EXPORT_GRID_FILE = "temp/grid.txt"
EXPORT_COORDS_FILE = "temp/coords.txt"  # Stores precise swipe coordinates

# Outer card scanning boundaries
SCAN_X_MIN = 20
SCAN_X_MAX = 1060
X_MIN = 42
GRID_WIDTH = 996
X_MAX = X_MIN + GRID_WIDTH

# =========================================
# Handling underscores in fun fact mode
# =========================================

# Replace the old parse_fun_fact_blanks with this clean text consolidator
def process_fun_fact_text(ocr_text_lines):
	"""
	Cleans up the raw EasyOCR lines from the fun fact area, preserving
	the readable words and structural spaces.
	"""
	# Join everything into a single line, making it uppercase for matching consistency
	full_text = " ".join(ocr_text_lines).strip().upper()
	# Replace any weird OCR artifacts from the underscores with spaces
	sanitized_text = re.sub(r"[._\-—~]{2,}", " ", full_text)
	# Reduce multiple spaces down to single spaces
	sanitized_text = re.sub(r"\s+", " ", sanitized_text)
	# print(f"DEBUG: Cleaned Text Template -> {sanitized_text}") # uncomment to debug
	return sanitized_text
	

# =====================================================================
# AUTO-DETECT GAME MODE
# =====================================================================
def detect_and_set_game_mode(ocr_results):
	"""
	Scans the top-level text elements returned by EasyOCR 
	to flag specialized level environments.
	"""
	combined_header_text = " ".join([res[1].upper() for res in ocr_results])
	
	if "FUN FACT" in combined_header_text or "FACT" in combined_header_text:
		print("🚨 SYSTEM: 'Fun Fact' banner detected! Shifting pipeline to Matrix-First Extraction.")
		return ["fun_fact", combined_header_text]
	
	if "OPPOSITE" in combined_header_text:
		print("🚨 SYSTEM: 'Antonyms' banner detected!")
		return ["antonym", combined_header_text]	
		
	if "SYNONYMS" in combined_header_text:
		print("🚨 SYSTEM: 'Synonyms' banner detected!")
		return ["synonym", combined_header_text]
	
	if "RHYMING" in combined_header_text:
		print("🚨 SYSTEM: 'Rhymes' banner detected!")
		return ["rhyme", combined_header_text]
	
	if "SEQUENCE" in combined_header_text:
		print("🚨 SYSTEM: 'Sequence' banner detected!")
		return ["sequence", combined_header_text]
	
	return ["standard", combined_header_text]


# =====================================================================
# STEP 1.1: ADB Screen Capture Function
# =====================================================================
def capture_screen(output_name=DEFAULT_IMAGE_NAME):
	"""Triggers ADB to capture the screen and directly saves it as a local PNG."""
	print(f"Taking screenshot via ADB and saving to {output_name}...")
	try:
		cmd = ["adb", "exec-out", "screencap", "-p"]
		with open(output_name, "wb") as f:
			subprocess.run(cmd, stdout=f, check=True)
		print("Screenshot captured successfully.")
		return True
	except FileNotFoundError:
		print("Error: ADB is not installed or not in your system PATH.")
		return False
	except subprocess.CalledProcessError as e:
		print(f"Error capturing screenshot: {e}")
		return False


# =====================================================================
# SQUARE-PADDED TEMPLATE MATCHING
# =====================================================================
def match_alphabet_templates(cell_gray_100x100):
	"""Compares 100x100 cell against templates. Returns matched char or None."""
	best_char = None
	best_score = -1.0
	threshold = 0.8

	for letter in string.ascii_uppercase:
		template_path = []
		max_val = []		
		for i in range(5): # creating many templates may help different resolution?
			template_path.append (os.path.join(TEMPLATE_DIR, f"{letter}{i}.png"))
			if not os.path.exists(template_path[i]):
				max_val.append(-1)
				continue
			
			template = cv2.imread(template_path[i], cv2.IMREAD_GRAYSCALE)
			template_resized = cv2.resize(template, (100, 100))
			
			result = cv2.matchTemplate(cell_gray_100x100, template_resized, cv2.TM_CCOEFF_NORMED)
			_, val, _, _ = cv2.minMaxLoc(result)		
			max_val.append(val)
					
		if max(max_val) > best_score:
			best_score = max(max_val)
			best_char = letter
		
		#print(template_path) # uncomment to debug
		#print(max_val) # uncomment to debug
	
	if best_score >= threshold:
		return best_char
	return None


# =====================================================================
# CELL LETTER STROKE EXTRACTION (Square Padding)
# =====================================================================
def crop_to_letter_bounds(cell_gray):
	"""Crops tightly to black strokes, then pads to square to preserve aspect ratio."""
	_, thresh = cv2.threshold(cell_gray, 180, 255, cv2.THRESH_BINARY_INV)
	contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	if contours:
		x_coords = []
		y_coords = []
		for cnt in contours:
			if cv2.contourArea(cnt) > 20: # overcome the area of the round corner
				x, y, w, h = cv2.boundingRect(cnt)
				x_coords.extend([x, x + w])
				y_coords.extend([y, y + h])
				
		if x_coords and y_coords:
			x_min, x_max = min(x_coords), max(x_coords)
			y_min, y_max = min(y_coords), max(y_coords)
			
			cropped = cell_gray[y_min:y_max, x_min:x_max]
			h, w = cropped.shape
			
			max_dim = max(h, w)
			pad_top = (max_dim - h) // 2
			pad_bottom = max_dim - h - pad_top
			pad_left = (max_dim - w) // 2
			pad_right = max_dim - w - pad_left
			
			square_cropped = cv2.copyMakeBorder(
				cropped, 
				pad_top, pad_bottom, pad_left, pad_right, 
				cv2.BORDER_CONSTANT, 
				value=255
			)
			return square_cropped
			
	return cell_gray


# =====================================================================
# DYNAMIC LAYOUT DETECTION (CONTOURS)
# =====================================================================
def detect_layout_regions(gray_img):
	"""Locates the Y boundaries of the white cards."""
	_, thresh = cv2.threshold(gray_img, 225, 255, cv2.THRESH_BINARY)
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	candidate_cards = []
	for cnt in contours:
		x, y, w, h = cv2.boundingRect(cnt)
		if 850 <= w:  
			candidate_cards.append((x, y, w, h))
			
	candidate_cards.sort(key=lambda rect: rect[1])
	
	word_bank_rect = None
	grid_rect = None

	if len(candidate_cards) >= 2:
		word_bank_rect = candidate_cards[0]
		grid_rect = candidate_cards[1] # counted from below to avoid detection error (?)
	elif len(candidate_cards) == 1:
		x, y, w, h = candidate_cards[0]
		if y < gray_img.shape[0] * 0.40:
			word_bank_rect = candidate_cards[0]
		else:
			grid_rect = candidate_cards[0]
			
	if not word_bank_rect:
		word_bank_rect = (X_MIN, int(gray_img.shape[0] * 0.15), GRID_WIDTH, int(gray_img.shape[0] * 0.18))
	if not grid_rect:
		grid_rect = (SCAN_X_MIN, 700, (SCAN_X_MAX - SCAN_X_MIN), 1200)
		
	print(f"Dimensions of word bank are {word_bank_rect} and of the grid are {grid_rect}.")	
	return word_bank_rect, grid_rect
		

# =====================================================================
# Detect length of hidden words using gold coin template
# =====================================================================
def detect_hidden_word_lengths(word_bank_img="temp/word_bank_img.png", template_path="templates/coin_template.png"):
	"""
	Detects gold coins in the word bank area, groups them into individual sequences,
	and returns a list of tuples: (x, y, length) representing their positions and letter counts.
	"""
	if not os.path.exists(template_path):
		print(f"⚠️ Template file '{template_path}' missing. Cannot auto-detect hidden word lengths!")
		return []

	gray_bank = cv2.imread(word_bank_img)
	template = cv2.imread(template_path)
	
	gray_bank = cv2.cvtColor(gray_bank, cv2.COLOR_BGR2GRAY)
	template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	w, h = template.shape[::-1]

	res = cv2.matchTemplate(gray_bank, template, cv2.TM_CCOEFF_NORMED)
	threshold = 0.8  
	loc = np.where(res >= threshold)

	detected_points = []
	for pt in zip(*loc[::-1]):  
		if not any(np.linalg.norm(np.array(pt) - np.array(p)) < w * 0.5 for p in detected_points):
			detected_points.append(pt)

	if not detected_points:
		return []

	detected_points.sort(key=lambda p: p[1])  
	rows = []
	for pt in detected_points:
		placed = False
		for r in rows:
			if abs(r[0][1] - pt[1]) < h * 0.5:
				r.append(pt)
				placed = True
				break
		if not placed:
			rows.append([pt])

	hidden_word_entries = []
	max_horizontal_gap = w * 1.5 

	for r in rows:
		r.sort(key=lambda p: p[0])  
		
		current_word_size = 1
		start_pt = r[0]
		for i in range(len(r) - 1):
			gap = r[i+1][0] - r[i][0]
			if gap <= max_horizontal_gap:
				current_word_size += 1
			else:
				if current_word_size > 1:
					hidden_word_entries.append((start_pt[0], start_pt[1], current_word_size))
				current_word_size = 1
				start_pt = r[i+1]
		
		if current_word_size > 1:
			hidden_word_entries.append((start_pt[0], start_pt[1], current_word_size))

	return hidden_word_entries


# =====================================================================
# Grid & Word Bank Extraction Function
# =====================================================================
def extract_game_data(image_path=DEFAULT_IMAGE_NAME):
	"""Loads image, locates elements, maps coordinates, and extracts letters."""
	if not os.path.exists(image_path):
		raise FileNotFoundError(f"Image '{image_path}' not found.")

	img = cv2.imread(image_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
	reader = easyocr.Reader(['en'], gpu=True) 
	allowed_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" 
	
	(wb_x, wb_y, wb_w, wb_h), (g_x, g_y, g_w, g_h) = detect_layout_regions(gray)

	# --- 1. EXTRACT WORD BANK OR FUN FACT BLANKS ---
	print("Extracting upper word card area...")
	word_bank_crop = gray[wb_y + int(wb_h * 0.1) : wb_y + wb_h, 0 : 1080] # cut off 10% on the top in case half box exist
	cv2.imwrite("temp/word_bank_img.png", word_bank_crop)		
	
	# detect game mode in the banner
	word_bank_banner = gray[max(0, wb_y - 50) : wb_y + 50, 0 : 1080] # extend the banner to be about 100 px wide
	# zoom banner for better dection
	scale_banner = 3	
	wbb_h, wbb_w = word_bank_banner.shape
	word_bank_banner = 	cv2.resize(word_bank_banner, (wbb_w * scale_banner, wbb_h * scale_banner))	
	cv2.imwrite("temp/word_bank_banner.png", word_bank_banner)	
	
	word_banner_result = reader.readtext(word_bank_banner) # list of character doesn't have to be capitals
	
	mode = detect_and_set_game_mode(word_banner_result)
	words = []

		
	# Detect words in the word bank	
	word_results = reader.readtext(
		word_bank_crop,			
		text_threshold=0.85,
		low_text=0.5, # reduce false positive	
		allowlist=allowed_chars, 
		detail=1)
	

	if mode[0] == "fun_fact":
		# Scale it before reading so not missing out the underscore
		wbc_h, wbc_w = word_bank_crop.shape
		word_bank_crop = cv2.resize(word_bank_crop, (wbc_w * scale_banner, wbc_h * scale_banner))	
		cv2.imwrite("temp/word_bank_img.png", word_bank_crop)	
		
		raw_fact_results = reader.readtext(word_bank_crop, detail=0)
		# Pass a list containing the structured context string as the 'words' target
		context_string = process_fun_fact_text(raw_fact_results)
		words = [f"CONTEXT:{context_string}"]

	else:
		all_bank_elements = []
		if word_results:
			first_box, first_text, _ = word_results[0]
			header_height_threshold = word_bank_crop.shape[0] * 0.1 # cutoff the title box, about 10% after cutting by 30 px above
			start_index = 1 if (first_box[2][1] < header_height_threshold) else 0
			
			for bbox, text, _ in word_results[start_index:]:
				clean_entry = text.upper().strip()
				tokens = clean_entry.split()
				if not tokens:
					continue
				# Estimate coordinate positions for individual words if a block contains multiple items
				start_x = bbox[0][0]
				end_x = bbox[1][0]
				approx_width = (end_x - start_x) / len(tokens)
				
				for i, token in enumerate(tokens):
					token_x = int(start_x + (i * approx_width))
					token_y = bbox[0][1]
					all_bank_elements.append((token_x, token_y, token))
		
		# Fetch coin clusters as (x, y, length)
		hidden_entries = detect_hidden_word_lengths("temp/word_bank_img.png")
		for h_x, h_y, h_len in hidden_entries:
			all_bank_elements.append((h_x, h_y, "*" * h_len))

		# Group the elements into horizontal rows based on Y coordinates
		all_bank_elements.sort(key=lambda item: item[1])
		bank_rows = []
		row_y_threshold = 25  # Pixel threshold to consider words on the same line
		
		for elem in all_bank_elements:
			placed = False
			for r in bank_rows:
				if abs(r[0][1] - elem[1]) < row_y_threshold:
					r.append(elem)
					placed = True
					break
			if not placed:
				bank_rows.append([elem])
				
		# Sort each row horizontally from left to right and compile the sequential word list
		for r in bank_rows:
			r.sort(key=lambda item: item[0])
			for elem in r:
				words.append(elem[2])

	# --- 2. GRID DIMENSION & EXTREME COORDINATES DETECTION ---
	print("Detecting grid layout and precise boundaries...")
	grid_crop = gray[g_y:g_y + g_h, X_MIN:X_MAX]
	scale = 4 # Zoom the grid big enough so EasyOCR can detect letters correctly	
	grid_crop_zoom = cv2.resize(grid_crop, (g_w * scale, g_h * scale))
	cv2.imwrite("temp/grid_crop_zoom.png", grid_crop_zoom) #uncomment this line to debug

	
	##### New method: Use cv2 to find exact grid coordinates
	### Find contours
	_, thresh = cv2.threshold(grid_crop_zoom, 120, 255, cv2.THRESH_BINARY_INV)
	contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)	


	# --- CONFIGURATION (BEFORE UPSCALE) ---
	MAX_CHAR_HEIGHT = 150 * scale
	MIN_CHAR_HEIGHT = 40 * scale
	MAX_CHAR_WIDTH = 150 * scale 
	ISOLATION_TARGET_HEIGHT = 40 *scale
	# Custom pipeline upscaling config
	
	# Filter contours and process in isolation
	all_x = []
	all_y = []

	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)

		# Apply your filters (ignore noises or full lines)
		if h <= MAX_CHAR_HEIGHT and w <= MAX_CHAR_WIDTH and h > MIN_CHAR_HEIGHT and w > 5:
			
			# --- CUSTOM PIPELINE: Begin Isolation ---
			
			# A. Crop to border with minimal 1px safety margin
			y_start = max(0, y - 1)
			y_end = min(thresh.shape[0], y + h)
			x_start = max(0, x - 1)
			x_end = min(thresh.shape[1], x + w)
			char_crop = img[y_start:y_end, x_start:x_end]
			
			cx = (x_end + x_start) / (2 * scale) # divide by scale to get the real coordinates
			cy = (y_end + y_start) / (2 * scale) # divide by scale to get the real coordinates
			all_x.append(cx)
			all_y.append(cy)

	##### Old method: use EasyOCR to help finding grid coordinates, problem: may detect false letters and thus mess up the grid	
	#for bbox, text, _ in valid_detections:
		# Calculate center of each character detection box relative to grid crop
	#	cx = (bbox[0][0] + bbox[1][0]) / (2 * scale) # divide by scale to get the real coordinates
	#	cy = (bbox[0][1] + bbox[2][1]) / (2 * scale) # divide by scale to get the real coordinates
	#	all_x.append(cx)
	#	all_y.append(cy)

	# Cluster to count ROW/COL
	def cluster_coordinates(coords, threshold = scale * 10):	 # reduce this number as the grid size grow?
		coords_sorted = sorted(coords)
		clusters = []
		if not coords_sorted:
			return 0
		current_cluster = [coords_sorted[0]]
		for val in coords_sorted[1:]:
			if val - current_cluster[-1] < threshold:
				current_cluster.append(val)
			else:
				clusters.append(current_cluster)
				current_cluster = [val]
		clusters.append(current_cluster)
		return len(clusters)

	COL = max(4, cluster_coordinates(all_x))
	ROW = max(6, cluster_coordinates(all_y))
	print(f"Detected a {ROW}x{COL} grid layout.")

	# Calculate exact active letter grid boundaries (ignoring outer margins)
	# This is relative to the cropped grid container
	active_x_min = min(all_x)
	active_x_max = max(all_x)
	active_y_min = min(all_y)
	active_y_max = max(all_y)

	# Calculate precise step sizes between character centers
	active_width = active_x_max - active_x_min
	active_height = active_y_max - active_y_min
	
	col_step = active_width / (COL - 1) if COL > 1 else active_width
	row_step = active_height / (ROW - 1) if ROW > 1 else active_height

	# --- 3. EXTRACT LETTER GRID & STORE ABSOLUTE CENTERS ---
	cell_w = GRID_WIDTH / COL
	cell_h = g_h / ROW
	
	grid = []
	coord_grid = [] # Stores (X, Y) absolute screen centers
	
	for r in range(ROW):
		row_letters = []
		row_coords = []
		for c in range(COL):
			# Calculate absolute screen coordinates for the center of the letter
			abs_center_x = int(X_MIN + active_x_min + (c * col_step))
			abs_center_y = int(g_y + active_y_min + (r * row_step))
			row_coords.append((abs_center_x, abs_center_y))

			# Crop bounds for detection
			x1 = int(c * cell_w)
			y1 = int(r * cell_h)
			x2 = int((c + 1) * cell_w)
			y2 = int((r + 1) * cell_h)
			
			cell_crop = grid_crop[y1:y2, x1:x2]
			#cv2.imwrite(f"temp/cell-crop-{r}-{c}.png", cell_crop) ### uncomment to debug
			tight_letter_crop = crop_to_letter_bounds(cell_crop)			
			
			# 1. Template Match
			cell_100x100 = cv2.resize(tight_letter_crop, (100, 100))
			# cv2.imwrite(f"temp/{r}-{c}.png", cell_100x100) ### uncomment to debug or to create template
			letter = match_alphabet_templates(cell_100x100)
			
			# 2. EasyOCR Fallback
			if letter is None:
				cell_200x200 = cv2.resize(tight_letter_crop, (200, 200))
				padded_cell = cv2.copyMakeBorder(
					cell_200x200, 20, 20, 20, 20, 
					cv2.BORDER_CONSTANT, value=255
				)
				char_result = reader.readtext(
					padded_cell, 
					text_threshold = 0.75,
					allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ", 
					detail=0
				)
				letter = char_result[0].strip() if char_result else "?"
				#letter = "?"
			row_letters.append(letter)
			
		grid.append(row_letters)
		coord_grid.append(row_coords)

	return words, grid, coord_grid, ROW, COL, mode


# =====================================================================
# Extract topic of the round
# =====================================================================
def extract_level_topic(image_path="temp/word_bank_img.png"):
	"""
	Automatically crops the green header bar and extracts the level topic text.
	Accepts either a file path string or an in-memory BGR numpy image.
	"""
	if isinstance(image_path, str):
		img = cv2.imread(image_path)
	else:
		img = image_path

	if img is None:
		print("❌ Error: Invalid image provided for topic extraction.")
		return ""

	h, w, _ = img.shape
	y_min, y_max = 0, int(0.25 * h)
	x_min, x_max = int(0.25 * w), int(0.75 * w)

	header_crop = img[y_min:y_max, x_min:x_max]

	gray = cv2.cvtColor(header_crop, cv2.COLOR_BGR2GRAY)
	resized = cv2.resize(gray, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
	_, thresholded = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

	if np.mean(thresholded[0:10, 0:10]) < 127:
		thresholded = cv2.bitwise_not(thresholded)

	reader = easyocr.Reader(['en'], gpu=True)
	results = reader.readtext(thresholded)

	detected_words = [res[1].strip().upper() for res in results if res[1].strip()]
	
	if detected_words:
		topic = " ".join(detected_words)
		topic = "".join(char for char in topic if char.isalnum() or char.isspace()).strip()
		print(f"🎯 Automatically detected level topic: '{topic}'")
		return topic
	
	print("⚠️ Could not detect topic text in header. Falling back to default.")
	return ""
	

# =====================================================================
# STEP 1.3: Data Export Function
# =====================================================================
def export_extracted_data(words, grid, coord_grid, bank_file=EXPORT_BANK_FILE, grid_file=EXPORT_GRID_FILE, coord_file=EXPORT_COORDS_FILE):
	"""Saves word bank, 2D layout grid, and absolute coordinate map to text files."""
	with open(bank_file, "w", encoding="utf-8") as f:
		for word in words:
			f.write(f"{word}\n")
		
	with open(grid_file, "w", encoding="utf-8") as f:
		for row in grid:
			f.write(" ".join(row) + "\n")
			
	with open(coord_file, "w", encoding="utf-8") as f:
		for r, row in enumerate(coord_grid):
			line_coords = [f"{x},{y}" for x, y in row]
			f.write(" ".join(line_coords) + "\n")
			
	print(f"Data successfully exported to:\n - {bank_file}\n - {grid_file}\n - {coord_file}")


# =====================================================================
# Execution Demo
# =====================================================================
if __name__ == "__main__":
	#if capture_screen():
		try:
			words, grid, coord_grid, ROW, COL, mode = extract_game_data(DEFAULT_IMAGE_NAME)
			
			print("\n=== EXTRACTION RESULT PACK ===")
			print(f"Parsed Sequence/Blanks Mask: {words}")
			
			print(f"\n=== SOLVED GRID ARRAY ({ROW}x{COL}) ===")
			for row in grid:
				print(" ".join(row))
			print()
			
			export_extracted_data(words, grid, coord_grid)
				
		except Exception as e:
			print(f"Error parsing captured game: {e}")
