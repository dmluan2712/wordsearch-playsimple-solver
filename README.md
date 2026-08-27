# WordSearch (PlaySimple) Solver

A Python CLI tool designed to automatically solve the Android game **WordSearch** by **PlaySimple**. 

The solver captures the device screen over ADB, processes the puzzle layout, generates solution swipe paths, and executes touch actions directly on your phone.

Optimized and tested for the **Samsung Galaxy S25** ($1080 \times 2340$ screen resolution).

---

## Repository Architecture

* **`image_processor.py`**: Takes a screenshot via ADB and parses the screen layout. It detects the target word bank, the letter grid, and the active game mode. It exports extracted data into `.txt` files for easy debugging.
* **`solver.py`**: The core algorithmic solver. Uses Depth-First Search (DFS) to find valid word paths across the grid and outputs corresponding ADB touch/swipe commands to a text file. Utilizes an external dictionary to support special game modes (synonyms, antonyms, rhymes, fun facts, and picture-based puzzles where words are implicit).
* **`play.py`**: The main CLI entry point. Coordinates image processing, runs `solver.py`, and dispatches generated ADB commands to automatically play the game on your device.

---

## Credits & AI Attribution

The codebase for this repository was primarily generated with **Google Gemini** under guided step-by-step development.

**Warning:** The initial prompts below were only as starting points. Arriving at the current working state required heavy follow-up extensive debugging, grid boundary adjustments, and manual code optimization.

### Initial Prompts Used

* **For `image_processor.py`**:
  > *"Write a Python script using OpenCV to crop and analyze a screenshot of a word search game on an Android phone (1080x2340). Detect the letter grid, extract the list of target words to find, and save the parsed text to a local .txt file for debugging."*

* **For `solver.py`**:
  > *"Write a Python script that uses Depth-First Search (DFS) to search for target words within a 2D matrix of letters (supporting horizontal, vertical, and diagonal directions). Convert the winning paths into ADB swipe coordinates and save them to a file. Include support for looking up synonyms/antonyms using a dictionary file when word lists are implicit."*

* **For `play.py`**:
  > *"Write a Python CLI script that orchestrates taking an ADB screenshot, running the image processing and solver modules, and executing the resulting ADB swipe commands in sequence."*

---

## To-Do / Future Improvements

- [ ] **`image_processor.py`**: Improve edge and grid detection when the puzzle background and grid tile colors are almost identical or lack clear visual separation.
- [ ] **`play.py`**: Implement seamless hands-free auto-play for consecutive levels so the script can proceed automatically when idle/AFK without waiting for a manual `Enter` key press.

---

## Requirements & Quick Start

1. **Prerequisites**:
   * Android device (Samsung S25 or $1080 \times 2340$ resolution) with **USB Debugging** enabled.
   * **ADB** installed and added to system `PATH`.
   * **Python 3.x** with OpenCV (`opencv-python`), NumPy, and required dependencies installed.

2. **Run**:
   ```bash
   python play.py
   ```
