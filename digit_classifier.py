from utils.puzzle_locator import locate_puzzle
from utils.cell_extractor import extract_cell
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from puzzle_solver import SudokuSolver
import numpy as np
import argparse
import imutils
import cv2
from tabulate import tabulate

# Construct parser for passed arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained digit classifier")
ap.add_argument("-i", "--image", required=True,
                help="path to input Sudoku puzzle image")
args = vars(ap.parse_args())

# Load the trained model
print("---------- [INFO] LOAD MODEL ----------")
model = load_model(args["model"])

# Load the input image from disk
print("---------- [INFO] LOAD IMAGE ----------")

image = cv2.imread(args["image"])

# Locate puzzle in given image
(puzzleImage, warped) = locate_puzzle(image)

# Initialize board to hold puzzle cells
board = np.zeros((9, 9), dtype="int")

# Assuming puzzle is a 9x9 grid (81 individual cells)
# Turn the warped image into a 9x9 grid
stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9

# List to store x,y coordinates of cells
cellLocs = []

# Loop over the grid locations
for y in range(0, 9):
    # initialize the current list of cell locations
    row = []
    for x in range(0, 9):
        # Find (x, y)-coordinates of the current cell
        startX = x * stepX
        startY = y * stepY
        endX = (x + 1) * stepX
        endY = (y + 1) * stepY

        # Add the (x, y)-coordinates to the cell locations list
        row.append((startX, startY, endX, endY))

        # Extract the cell from the cropped image
        cell = warped[startY:endY, startX:endX]
        cell = extract_cell(cell)

        if cell is not None:
            # Resize the cell to 28x28 pixels
            cell_val = cv2.resize(cell, (28, 28))
            cell_val = cell_val.astype("float") / 255.0
            cell_val = img_to_array(cell_val)
            cell_val = np.expand_dims(cell_val, axis=0)

            # Classify the cell value
            pred = model.predict(cell_val).argmax(axis=1)[0]
            board[y, x] = pred

        # Add the row to our cell locs
        cellLocs.append(row)

# Board with no zzeros for missing values
board_with_no_zeros = [["" if cell == 0 else cell for cell in row]
                       for row in board.tolist()]

# Solve the Sudoku puzzle with custom Z3 solver
print("----------- [INFO] SOLVING SUDOKU PUZZLE -----------")
solution = SudokuSolver(board.tolist()).solve()
print("9x9 PUZZLE")
print(tabulate(board_with_no_zeros, tablefmt="fancy_grid"))
if solution is None:
    print("NO SOLUTION FOUND")
else:
    print("SOLUTION FOUND")
    print(tabulate(solution, tablefmt="fancy_grid"))
