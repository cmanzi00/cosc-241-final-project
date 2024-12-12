# CNN Trained Digit Classification Model w/ Sudoku Solver

This project trains a CNN model to classify digits, enabling it to extract Sudoku puzzles from images and solve them programmatically using the Z3 Solver. 

---

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation & Usage](#installation--usage)
- [Sample](#sample)
- [Acknowledgments](#acknowledgments)

---

## Prerequisites

### System Requirements
- **Python**: Version 3.8 - 3.11
- **Pip**: Version >19.0 (or >20.3 for macOS)

### Required Libraries
Install the necessary dependencies:
```sh
pip install tensorflow numpy opencv-contrib-python scikit-image tabulate imutils scikit-learn matplotlib progressbar2 pandas
```

---

## Installation & Usage

### 1. Clone the Repository
```sh
git clone https://github.com/cmanzi00/cosc-231-final-project.git
cd cosc-231-final-project
```

### 2. Train the Model
Train the CNN model and save it with your desired name:
```sh
python trainer.py --model output/{model_name}
```

### 3. Solve a Sudoku Puzzle
Provide the trained model and an image of the puzzle to solve it:
```sh
python puzzle_solver.py --model output/{model_name} --image images/{image}
```
Ensure the puzzle image is stored in the `images` folder (the `images` folder contains some test images that can be used).

---

## Sample
Include visual demonstrations (e.g., screenshots, GIFs, or results).

---

## Acknowledgments
- [Grid Detection and Cell Extraction - StackOverflow](https://stackoverflow.com/questions/59182827)
- [Paper on Hyperparameter Tuning for High Accuracy](https://www.mdpi.com/741864)
- [Sudoku Images Repo](https://github.com/kirkeaton/sudoku-image-solver/tree/main/sudoku_images)
