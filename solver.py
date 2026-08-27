import sys, os, time
import urllib.request
import json
import urllib.parse
import difflib
import numpy as np
import re
from nltk.corpus import wordnet as wn
from image_processor import capture_screen, extract_game_data, export_extracted_data

# =====================================================
# Auto-correct function to avoid false word recognition
# =====================================================


def levenshtein_distance(s1, s2):
	"""Calculates the minimum edit distance between two strings."""
	if len(s1) < len(s2):
		return levenshtein_distance(s2, s1)
	if len(s2) == 0:
		return len(s1)

	previous_row = range(len(s2) + 1)
	for i, c1 in enumerate(s1):
		current_row = [i + 1]
		for j, c2 in enumerate(s2):
			insertions = previous_row[j + 1] + 1
			deletions = current_row[j] + 1
			substitutions = previous_row[j] + (c1 != c2)
			current_row.append(min(insertions, deletions, substitutions))
		previous_row = current_row

	return previous_row[-1]

def auto_correct_ocr_typos(sequence, dictionary, pool_size = 3):
	"""
	Uses Levenshtein Distance to evaluate OCR typos. Compiles all valid 
	words, placeholders, and multi-word candidates into a single flat list.
	"""
	flat_corrected_sequence = []
	clean_dict = {w.strip().lower() for w in dictionary}
	
	for entry in sequence:
		# Pass placeholders straight through
		if "*" in entry:
			flat_corrected_sequence.append(entry)
			continue
			
		# Skip purely non-alphabetic strings or spaces
		if not entry.isalpha():
			continue
			
		lookup_word = entry.strip().lower()
		
		# If it's already a valid dictionary word, append it directly
		if lookup_word in clean_dict:
			flat_corrected_sequence.append(entry.upper())
			continue
			
		# If it's a typo, gather the top matching suggestions
		candidates = []
		target_len = len(lookup_word)
		
		for dict_word in clean_dict:
			if abs(len(dict_word) - target_len) > 2:
				continue
				
			dist = levenshtein_distance(lookup_word, dict_word)
			if dist <= 2:
				candidates.append((dist, dict_word.upper()))
				
		# Sort primarily by lowest distance, then alphabetically
		candidates.sort(key=lambda x: (x[0], x[1]))
		top_suggestions = [word for dist, word in candidates[:pool_size]]
		
		if top_suggestions:
			# print(f"🔧 OCR Levenshtein Pool: Expanding typo '{entry}' into candidates: {top_suggestions}") ### uncomment this to debug
			# Extend the single flat list with all candidate variations
			flat_corrected_sequence.extend(top_suggestions)
		else:
			# Fallback to the original string if no close dictionary words match
			flat_corrected_sequence.append(entry.upper())
			
	return flat_corrected_sequence

# =================================================
# Datamuse Antonym Fetcher to handle Twister Mode
# ==================================================
'''
This function takes a word from your word bank and queries Datamuse for its clean English antonyms:
'''

def fetch_antonyms(word, max_param=1000):
	"""
	Queries Datamuse API for strict antonyms (opposites) of a given word.
	"""
	formatted_word = word.strip().lower()
	url = f"https://api.datamuse.com/words?rel_ant={urllib.parse.quote(formatted_word)}&max={max_param}"
	
	antonyms = set()
	try:
		req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
		with urllib.request.urlopen(req, timeout=4) as response:
			data = json.loads(response.read().decode())
			for item in data:
				ant_word = item['word'].upper()
				if ant_word.isalpha() and len(ant_word)>2:
					antonyms.add(ant_word)
	except Exception as e:
		print(f"   ⚠️ Antonym fetch failed for {word}: {e}")
		
	return antonyms

'''
This function takes a word from your word bank and queries Datamuse for its clean English synonyms:
'''
def fetch_synonyms(word, max_param=1000):
	"""
	Queries Datamuse API for strict synonyms (means-like) of a given word.
	"""
	formatted_word = word.strip().lower()
	url = f"https://api.datamuse.com/words?ml={urllib.parse.quote(formatted_word)}&max={max_param}"
	
	synonyms = set()
	try:
		req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
		with urllib.request.urlopen(req, timeout=4) as response:
			data = json.loads(response.read().decode())
			for item in data:
				syn_word = item['word'].upper()
				if syn_word.isalpha() and len(syn_word)>2:
					synonyms.add(syn_word)
	except Exception as e:
		print(f"   ⚠️ Synonym fetch failed for {word}: {e}")
		
	return synonyms

'''
This function takes a word from your word bank and queries Datamuse for its rhymes:
'''
def fetch_rhymes(word,max_param=1000):
	"""
	Queries Datamuse API for perfect rhymes (rel_rhy) of a given word.
	"""
	formatted_word = word.strip().lower()
	url = f"https://api.datamuse.com/words?rel_rhy={urllib.parse.quote(formatted_word)}&max={max_param}"
	
	rhymes = set()
	try:
		req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
		with urllib.request.urlopen(req, timeout=8) as response:
			data = json.loads(response.read().decode())
			for item in data:
				rhy_word = item['word'].upper()
				if rhy_word.isalpha():
					rhymes.add(rhy_word)
	except Exception as e:
		print(f"   ⚠️ Rhyme fetch failed for {word}: {e}")
	
	# LOCAL FALLBACK ENGINE:
	# If the word ends with common high-volume suffixes, find local dictionary matches of the same length
	word_upper = word.upper()
	common_suffixes = ["ILL", "OW", "OUGH", "IGHT", "AD", "ANK", "OUT", "AY", "ET", "EE", "EAR", "EA"]
	
	for suffix in common_suffixes:
		if word_upper.endswith(suffix) and len(word_upper) >= len(suffix):
			# Load your cached dictionary strings locally
			with open("templates/english_words.txt", "r", encoding="utf-8") as f:
				local_words = {line.strip().upper() for line in f if line.strip()}
			
			# Filter for words matching the length and the same ending block
			target_len = len(word_upper)
			local_matches = {w for w in local_words if len(w) == target_len and w.endswith(suffix)}
			
			rhymes.update(local_matches)
			print(f"🎯 Local Rhyme Engine infused {len(local_matches)} local words ending in '-{suffix.lower()}'")
			break
			
	return rhymes
		
# ==================
# LOCAL NLTK LIBRARY
# ==================
def fetch_synonyms_nltk(word: str, max_depth: int = 3) -> set:
	"""
	Fetches synonyms for a given word using NLTK WordNet.
	
	:param word: Target word to query.
	:param max_depth: Level of semantic expansion/bridging (1 = direct synonyms).
	:return: A set of uppercase synonym strings.
	"""
	synonyms = set()
	target_word = word.lower().strip()
	
	visited_synsets = set()
	current_words = {target_word}
		
	# fast_dict, deep_dict = load_two_tier_dictionaries()

	for depth in range(max_depth):
		next_words = set()
		
		for w in current_words:
			for synset in wn.synsets(w):
				if synset in visited_synsets:
					continue
				visited_synsets.add(synset)

				for lemma in synset.lemmas():
					lemma_name = lemma.name().replace('_', ' ').upper()
					
					# Exclude the original query word from the result set
					if lemma_name.lower() != target_word:
						synonyms.add(lemma_name)
						# Feed clean single-token words into the next depth layer
						if ' ' not in lemma_name and len(lemma_name)>2:
							next_words.add(lemma_name.lower())
						# synonyms.union( fetch_synonyms(lemma_name.lower(), 20)) ### uncomment for even more word choice
							
		current_words = next_words

	return synonyms


def fetch_antonyms_nltk(word: str, max_depth: int = 3) -> set:
	"""
	Fetches antonyms for a given word using NLTK WordNet.
	Uses depth to find direct antonyms (depth=1) or antonyms of 
	bridged synonyms (depth > 1).
	
	:param word: Target word to query.
	:param max_depth: Level of semantic expansion/bridging.
	:return: A set of uppercase antonym strings.
	"""
	antonyms = set()
	target_word = word.lower().strip()

	# fast_dict, deep_dict = load_two_tier_dictionaries()

	# Step 1: Expand target word to include its synonyms up to (max_depth - 1)
	# Depth 1 directly uses target_word; Depth 2 includes target_word's direct synonyms, etc.
	base_words = {target_word}
	if max_depth > 1:
		base_words.update({s.lower() for s in fetch_synonyms_nltk(target_word, max_depth=max_depth - 1)})

	# Step 2: Find direct antonyms for all gathered base words
	antonym_seed_words = set()
	for b_word in base_words:
		for synset in wn.synsets(b_word):
			for lemma in synset.lemmas():
				for ant in lemma.antonyms():
					ant_name = ant.name().replace('_', ' ').upper()
					if ant_name.lower() != target_word:
						antonyms.add(ant_name)
						if ' ' not in ant_name  and len(ant_name)>2:
							antonym_seed_words.add(ant_name.lower())

	# Step 3: If depth remains, expand the antonyms themselves via synonym bridging
	# E.g., for depth=2, this gets synonyms of the found antonyms
	if max_depth > 1 and antonym_seed_words:
		for ant_word in antonym_seed_words:
			bridged_antonyms = fetch_synonyms_nltk(ant_word, max_depth=max_depth - 1)
			# bridged_antonyms = bridged_antonyms.union( fetch_synonyms(ant_word, 20)) ### uncomment for even more word choice
			antonyms.update(bridged_antonyms)

	# Clean up to ensure original target word isn't returned
	antonyms.discard(word.upper())
	return antonyms
	
def fetch_category_nltk(category: str, max_depth: int = 5) -> set:
	"""
	Extracts category members (hyponyms) using NLTK WordNet.
	Recursively checks child synsets up to `max_depth`.
	"""
	results = set()
	
	# Get all noun synsets matching the category word
	synsets = wn.synsets(category.lower(), pos=wn.NOUN)
	if not synsets:
		return results

	def get_hyponyms(synset, current_depth):
		if current_depth > max_depth:
			return
		for hyponym in synset.hyponyms():
			for lemma in hyponym.lemmas():
				# Clean up multi-word entries (e.g., 'rocking_chair' -> 'ROCKING CHAIR' or 'ROCKINGCHAIR')
				word = lemma.name().replace('_', ' ').upper()
				if len(word)>2:
					results.add(word)
				#results.union( fetch_synonyms(word,50)) ### uncomment to get even more words
			# Recurse down the hierarchy tree
			get_hyponyms(hyponym, current_depth + 1)
	
	for syn in synsets:
		get_hyponyms(syn, current_depth=1)

	return results

# =====================================================================
# CONFIGURATION & CONSTANTS
# =====================================================================
INPUT_BANK_FILE = "temp/bank.txt"
INPUT_GRID_FILE = "temp/grid.txt"
INPUT_COORDS_FILE = "temp/coords.txt"
OUTPUT_COMMANDS_FILE = "temp/swipe_commands.txt"
LOCAL_DICT_FILE = "templates/english_words.txt"

# Standard Scrabble-style dictionary containing inflections, plurals, and past tenses
DICT_URL = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"

DIRECTIONS = [
	(-1, -1), (-1, 0), (-1, 1),  # Up-Left, Up, Up-Right
	(0, -1),		   (0, 1),   # Left, Right
	(1, -1),  (1, 0),  (1, 1)	# Down-Left, Down, Down-Right
]

# =========================
# DYNAMIC GRID WORDS FINDER
# =========================

# The solver checks the ultra-clean fast list first. 
# If it finds enough valid candidates to satisfy the board requirements,
# it skips the deep lookup entirely.

def load_two_tier_dictionaries(path_to_fast_dictionary = "templates/top_english.txt", path_to_slow_dictionary = "templates/english_words.txt"):
	"""
	Loads a fast, clean core vocabulary list and a comprehensive 
	backup dictionary for deep semantic bridging.
	"""
	# 1. Fast-Pass List: Top 10k common, actual words (No abbreviations)
	with open(path_to_fast_dictionary, "r", encoding="utf-8") as f:
		fast_dict = {line.strip().lower() for line in f if len(line.strip()) >= 3}
		
	# 2. Deep-Search List: The massive backup dictionary
	with open(path_to_slow_dictionary, "r", encoding="utf-8") as f:
		deep_dict = {line.strip().lower() for line in f if line.strip()}
		
	return fast_dict, deep_dict


# Finding all words on the grid
# using dynamic search
	
def extract_all_valid_grid_words(grid, r_max, c_max, fast_dict, deep_dict):
	"""
	Crawls the grid. Prioritizes the ultra-clean fast_dict to prevent 
	false positives, falling back to deep_dict only if a structural gap exists.
	"""
	directions = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]
	found_words = {}
	
	for r in range(r_max):
		for c in range(c_max):
			for dr, dc in directions:
				current_word = ""
				path = []
				curr_r, curr_c = r, c
				
				while 0 <= curr_r < r_max and 0 <= curr_c < c_max:
					if grid[curr_r][curr_c] == "?":
						break
						
					current_word += grid[curr_r][curr_c].strip().upper()
					path.append((curr_r, curr_c))
					
					if len(current_word) >= 3:
						word_lower = current_word.lower()
						
						# Tiebreaker logic: give massive weight to natural words
						if word_lower in fast_dict:
							found_words[current_word] = {"path": list(path), "priority": "high"}
						elif word_lower in deep_dict:
							# Only keep obscure words if they haven't been overridden by a clean word
							if current_word not in found_words:
								found_words[current_word] = {"path": list(path), "priority": "low"}
							
					curr_r += dr
					curr_c += dc
					
	return found_words
# =====================================================================
# 1. DICTIONARY & GAME STATE LOADERS
# =====================================================================
def ensure_local_dictionary():
	"""Ensures a comprehensive English dictionary is cached locally."""
	if not os.path.exists(LOCAL_DICT_FILE):
		print("📥 First-time setup: Downloading comprehensive English dictionary...")
		try:
			req = urllib.request.Request(DICT_URL, headers={'User-Agent': 'Mozilla/5.0'})
			with urllib.request.urlopen(req, timeout=10) as response:
				content = response.read().decode('utf-8')
				# Save locally for rapid subsequent loads
				with open(LOCAL_DICT_FILE, "w", encoding="utf-8") as f:
					f.write(content)
			print("💾 Dictionary successfully cached locally.")
		except Exception as e:
			raise RuntimeError(f"Failed to download reference dictionary: {e}")

	# Load into memory as a highly efficient lookup set
	with open(LOCAL_DICT_FILE, "r", encoding="utf-8") as f:
		return {line.strip().upper() for line in f if line.strip()}


def load_game_state_sequence():
	"""
	Reads the word list sequence directly, preserving the exact layout order
	so we can calculate alphabetical boundary constraints.
	"""
	if not (os.path.exists(INPUT_BANK_FILE) and os.path.exists(INPUT_GRID_FILE) and os.path.exists(INPUT_COORDS_FILE)):
		raise FileNotFoundError("Missing files! Please run image_processor.py first.")

	# Read the layout sequence exactly as extracted by OCR (Left-to-Right, Top-to-Bottom)
	raw_sequence = []
	with open(INPUT_BANK_FILE, "r", encoding="utf-8") as f:
		for line in f:
			val = line.strip().upper()
			if val:
				raw_sequence.append(val)

	# Load grid matrix
	grid = []
	with open(INPUT_GRID_FILE, "r", encoding="utf-8") as f:
		for line in f:
			if line.strip():
				grid.append(line.strip().split())

	# Load screen coordinates mapping
	coords = []
	with open(INPUT_COORDS_FILE, "r", encoding="utf-8") as f:
		for line in f:
			if line.strip():
				row_coords = []
				for pair in line.strip().split():
					x, y = map(int, pair.split(","))
					row_coords.append((x, y))
				coords.append(row_coords)

	return raw_sequence, grid, coords


# =====================================================================
# 2. ALPHABETICAL BOUNDARY COMPUTER
# =====================================================================
def get_alphabetical_bounds(sequence, index):
	"""
	Finds the closest known words to the left and right of the hidden word index
	to calculate strict alphabetical bounds.
	"""
	left_bound = None
	right_bound = None

	# Find the closest known word to the left
	for i in range(index - 1, -1, -1):
		if all(c.isalpha() for c in sequence[i]): # and "COIN" not in sequence[i]:
			left_bound = sequence[i]
			break

	# Find the closest known word to the right
	for i in range(index + 1, len(sequence)):
		if all(c.isalpha() for c in sequence[i]): # and "COIN" not in sequence[i]:
			right_bound = sequence[i]
			break

	return left_bound, right_bound


# =====================================================================
# 3. PATH-FINDERS & GRID STRING HARVESTER
# =====================================================================
def find_word_path(word, grid, r_max, c_max):
	"""Finds a valid straight line path for a specific target word."""
	word_len = len(word)
	if word_len>2:
		for r in range(r_max):
			for c in range(c_max):
				if grid[r][c] != word[0]:
					continue
				for dr, dc in DIRECTIONS:
					end_r = r + dr * (word_len - 1)
					end_c = c + dc * (word_len - 1)
					if 0 <= end_r < r_max and 0 <= end_c < c_max:
						if grid[end_r][end_c] == word[-1]:
							match_found = True
							potential_path = []
							for step in range(word_len):
								curr_r = r + dr * step
								curr_c = c + dc * step
								if grid[curr_r][curr_c] != word[step]:
									match_found = False
									break
								potential_path.append((curr_r, curr_c))
							if match_found:
								return potential_path
	return None


def harvest_grid_strings_by_length(grid, length, r_max, c_max):
	"""Generates all physical straight-line sequences of length L hidden in the grid."""
	candidates = {}
	for r in range(r_max):
		for c in range(c_max):
			for dr, dc in DIRECTIONS:
				end_r = r + dr * (length - 1)
				end_c = c + dc * (length - 1)
				if 0 <= end_r < r_max and 0 <= end_c < c_max:
					chars = [grid[r + dr * i][c + dc * i] for i in range(length)]
					word = "".join(chars)
					path = [(r + dr * i, c + dc * i) for i in range(length)]
					candidates[word] = path
	return candidates

# ========================
# 4.1 FUN FACT MODE SOLVER
# ========================
def solve_fun_fact_level(words, grid, r_max, c_max):
	"""
	Solves Fun Fact levels by crawling the grid for real words and filtering
	them accurately against a heavily sanitized context token list.
	"""
	# Load your local dictionary configurations
	fast_dict, deep_dict = load_two_tier_dictionaries("templates/top_english.txt","templates/english_words.txt")

	context_str = words[0].replace("CONTEXT:", "").upper()
	print(f"🧩 Solving Fun Fact Context: '{context_str}'")
	
	# Step 1: Run the comprehensive Matrix-First Extraction
	detected_grid_words = extract_all_valid_grid_words(grid, r_max, c_max, fast_dict, deep_dict)
	
	# Step 2: Clean up the sentence text aggressively using Regex.
	# This turns "ANDALUSIA'S" into ["ANDALUSIA", "S"] and strips numbers like "3)"
	known_sentence_words = set(re.findall(r'[A-Z]{2,}', context_str)) ### uncomment to debug
	
	# Step 3: Match candidates
	candidate_solutions = {}
	for word, path in detected_grid_words.items():
		### Uncomment to see if it's true: Only process if the word found on the grid isn't printed out in the clue sentence
		#if word not in known_sentence_words:
			candidate_solutions[word] = path
	
	print(f"💡 Found {len(candidate_solutions)} candidate words physically on the grid.")

	solved_paths = {}
	for word, path in candidate_solutions.items():
		if path["priority"] == "high":
			print(f"  ✅ Match Found: {word}")
			solved_paths[word] = find_word_path(word, grid, r_max, c_max)
		if path["priority"] == "low" and len(word) > 4: # also find words in past tense or plurals, but longer than 4 letters
			print(f"  ✅ Match Found: {word}")
			solved_paths[word] = find_word_path(word, grid, r_max, c_max)

	return solved_paths


# ========================
# 4.2 PICTURE MODE SOLVER
# ========================
def solve_picture_level(mode, grid, r_max, c_max):
	
	# Load your local dictionary configurations
	fast_dict, deep_dict = load_two_tier_dictionaries("templates/top_english.txt", "templates/english_words.txt")
	
	solved_paths = {}
	
	banner_str = mode[1].upper() # find all words in the banner
	banner_str = set(re.findall(r'[A-Z]{2,}', banner_str)) #Clean up the sentence text aggressively using Regex
	print(f"🧩 Solving Picture by finding category words of: {banner_str}")
	
	# Step 1: Run the comprehensive Matrix-First Extraction
	detected_grid_words = extract_all_valid_grid_words(grid, r_max, c_max, fast_dict, deep_dict)
	
	# Step 2: Find synonyms/ category words for each word in the title
	
	for entry in banner_str:
		if entry.isalpha() and len(entry)>1:
			target_pool = fetch_synonyms(entry).union( fetch_category_nltk(entry) ) # merge two sets of synonyms of each word in the banner
			for target_word in target_pool:
				path = find_word_path(target_word, grid, r_max, c_max)
				if path:
					print(f"  ✅ Match: '{target_word}'")
					solved_paths[target_word] = path
	
	# Step 3: Heuristic Brute force and find all words in the small dictionary:
	candidate_solutions = {}
	for word, path in detected_grid_words.items():
		candidate_solutions[word] = path				
	
	print(f"💡 Found {len(candidate_solutions)} candidate words physically on the grid.")

	for word, path in candidate_solutions.items():
		if path["priority"] == "high":
			print(f"  ✅ Match Found: {word}")
			solved_paths[word] = find_word_path(word, grid, r_max, c_max)
		if path["priority"] == "low" and len(word) > 4: # also find words in past tense or plurals, but longer than 4 letters
			print(f"  ✅ Match Found: {word}")
			solved_paths[word] = find_word_path(word, grid, r_max, c_max)

	return solved_paths

# =====================================================================
# 4.3 SOLVER LOGIC (ALPHABETICAL BRUTE FORCE)
# =====================================================================
def solve_level_with_brute_force(raw_sequence, grid, r_max, c_max, mode):
	"""
	Processes the exact layout sequence. Known words are solved directly.
	Hidden words are resolved using alphabetical boundary brute-forcing.
	"""
	dictionary = ensure_local_dictionary()
	
	fast_dict, deep_dict = load_two_tier_dictionaries()

	# Run the comprehensive Matrix-First Extraction
	detected_grid_words = extract_all_valid_grid_words(grid, r_max, c_max, fast_dict, deep_dict) ### uncomment to debug	
	# print(detected_grid_words) ### uncomment to debug	
	
	# We will wrap the execution loop in a helper so we can easily pivot/restart
	#def execute_solving_run(raw_sequence, mode):
	
	"""
	Processes the word bank sequence with auto-correction for OCR anomalies.
	"""
	
	solved_paths = {}
		
	# Separate out all cleanly visible known words for substring validation
	
	mode_label = "[TWISTER MODE: fun fact]" if mode[0] == "fun_fact" else (
		"[TWISTER MODE: antonyms]" if mode[0] == "antonym"	else (
		"[TWISTER MODE: synonyms]" if mode[0] == "synonym" else (
		"[TWISTER MODE: rhymes]" if mode[0] == "rhyme" else (
		"[PICTURE MODE]" if len(raw_sequence) < 2 else(  # too few words means false alarm 
		"[SEQUENCE MODE]" if mode[0]=="sequence" else "[STANDARD MODE]")))))

	print(f"--- Solving Sequence {mode_label} ---")

	if mode_label == "[PICTURE MODE]":
		return solve_picture_level(mode, grid, r_max, c_max)	

	if mode[0] == "fun_fact":
		return solve_fun_fact_level(raw_sequence, grid, r_max, c_max)	
	
	if mode[0] == "sequence":
		solved_paths = solve_fun_fact_level(raw_sequence, grid, r_max, c_max)
		# Also, check all words in the box itself
		for word in raw_sequence:
			path = find_word_path(word, grid, r_max, c_max)
			if path:	
				print(f"  ✅ Match Found: {word}")
				solved_paths[word] = path
		return solved_paths

	# Run the auto-correct filter first!
	sequence = auto_correct_ocr_typos(raw_sequence, dictionary)	

	for idx, entry in enumerate(sequence):
		# Handle gold-coin lines
		if "*" in entry:
			length = len(entry)	
			left_bound, right_bound = get_alphabetical_bounds(sequence, idx)
			
			
			known_words = [item for item in sequence if item.isalpha() ] # and "COIN" not in item]
	

			# Diagnostic boundary reporting. Uncomment to debug
			bound_str = f"Alphabetical constraints: "
			bound_str += f"'{left_bound}' <= WORD" if left_bound else "START <= WORD"
			bound_str += f" <= '{right_bound}'" if right_bound else " <= END"
			print(f"\n🪙 Hidden Word (Length {length}) found at index {idx}. {bound_str}")

			# 1. Gather all strings of this exact length on the grid
			grid_possibilities = harvest_grid_strings_by_length(grid, length, r_max, c_max)
			valid_solutions = []

			# 2. Filter using alphabetical, substring, and English dictionary tests
			for candidate, path in grid_possibilities.items():
				# Test 1: Must be a legitimate English word (allows plurals and tenses)
				if candidate not in dictionary:
					continue

				# Test 2: Cannot be an exact duplicate of an already solved word
				if candidate in solved_paths:
					continue

				# Test 3: Must respect alphabetical left boundary
				#if left_bound and candidate < left_bound:
				#	continue

				# Test 4: Must respect alphabetical right boundary
				#if right_bound and candidate > right_bound:
				#	continue

				# Test 5: Cannot be a substring of any known word (e.g. "EACH" is out if "TEACH" is a word)
				is_substring_of_known = False
				for known in known_words:
					if candidate in known:
						is_substring_of_known = True
						break
				if is_substring_of_known:
					continue

				# If it passes all strict filters, it's a qualified candidate!
				print(f"Candidate {candidate} is found")
				valid_solutions.append((candidate, path))

			# 3. Resolve the path
			if len(valid_solutions) == 1:
				solved_word, path = valid_solutions[0]
				print(f"   ✨ Alphabetical match resolved: '{solved_word}'!")
				solved_paths[solved_word] = path
			elif len(valid_solutions) > 1:
				# If multiple valid words exist in the grid meeting the criteria, swipe them all!
				print(f"   💡 Multiple viable matches found: {[w for w, _ in valid_solutions]}. Swiping all.")
				for solved_word, path in valid_solutions:
					solved_paths[solved_word] = path
			#else:
				# print("   ⚠️ No valid English words matching these boundaries exist in the grid.") ### uncomment to debug
		
		# Handle the case where all letters are revealed
		elif entry.isalpha(): # and "COIN" not in entry:		

			# Stage-specific word expansion mapping
			if mode[0] == "antonym":
				target_pool = fetch_antonyms(entry).union( fetch_antonyms_nltk(entry) ) # merging two sets
				#print(f"{entry}: {target_pool}") ### uncomment to debug
			elif mode[0] == "synonym":
				target_pool = fetch_synonyms(entry).union( fetch_synonyms_nltk(entry) ) # merging two sets
				#print(f"{entry}: {target_pool}") ### uncomment to debug
			elif mode[0] == "rhyme":
				target_pool = fetch_rhymes(entry)
				#print(f"{entry}: {target_pool}") ### uncomment to debug
			else:
				target_pool = {entry}
		
			# Make sure the pool at least contains the fallback token
			if not target_pool and len(entry)>2:
				target_pool = {entry}
			
			# Scan the physical grid matrix
			for target_word in target_pool:
				path = find_word_path(target_word, grid, r_max, c_max)
				if path:
					print(f"  ✅ Match: '{target_word}'")
					solved_paths[target_word] = path
	
	if mode[0] in ["synonym", "antonym"]:	
		candidate_solutions = {}		
		for word, path in detected_grid_words.items():
			candidate_solutions[word] = path				

		for word, path in candidate_solutions.items():
			if path["priority"] == "high" and len(word) >=3 : # find words that are at least 3 letters
				print(f"  ✅ Match Found: {word}")
				solved_paths[word] = find_word_path(word, grid, r_max, c_max)

			if path["priority"] == "low" and len(word) >= 4: # also find words in past tense or plurals, but longer than 4 letters
				print(f"  ✅ Match Found: {word}")
				solved_paths[word] = find_word_path(word, grid, r_max, c_max)

	return solved_paths



# =====================================================================
# 5. SWIPE COMMAND COMPILER
# =====================================================================
def generate_adb_swipe_commands(word_paths, coords_map):
	adb_commands = []
	for word, path in word_paths.items():
		if not path or len(path) < 2:
			continue
		x1, y1 = coords_map[path[0][0]][path[0][1]]
		x2, y2 = coords_map[path[-1][0]][path[-1][1]]
		
		command_string = f"# Word: {word}\n"
		command_string += f"adb shell input swipe {x1} {y1} {x2} {y2} 200\n"
		adb_commands.append(command_string)
	return adb_commands


if __name__ == "__main__":
	try:
		sequence, grid, coords = load_game_state_sequence()
		ROW_MAX = len(grid)
		COL_MAX = len(grid[0]) if ROW_MAX > 0 else 0
		
		solved_word_paths = solve_level_with_brute_force(sequence, grid, ROW_MAX, COL_MAX, mode)
		commands = generate_adb_swipe_commands(solved_word_paths, coords)
		
		with open(OUTPUT_COMMANDS_FILE, "w", encoding="utf-8") as f:
			f.write("#!/bin/bash\n\n")
			for block in commands:
				f.write(block + "\n")
				
		print(f"🚀 Swipe macro generated successfully!")
	except Exception as e:
		print(f"Execution failed: {e}")










