from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2


def locate_puzzle(image):

    # Resize the image
    image = imutils.resize(image, width=600)

    # Grayscale and blur the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # Apply adaptive thresholding and inversion to the threshold map
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # Find contours and sort them by size in desc order
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Initialize a contour corresponding to the puzzle outline
    puzzleContour = None

    # Loop over the contours
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # If the approx contour has four points, assume it is the puzzle outline
        if len(approx) == 4:
            puzzleContour = approx
            break

    # Raise an error if no puzzle found
    if puzzleContour is None:
        raise Exception(("Could not find Sudoku puzzle outline. "
                         "Try debugging your thresholding and contour steps."))

    # Apply a four point perspective transform to obtain a top-down bird's eye view of the puzzle
    puzzle = four_point_transform(
        image, puzzleContour.reshape(4, 2))
    warped = four_point_transform(gray, puzzleContour.reshape(4, 2))

    # Return puzzle in both RGB and grayscale
    return (puzzle, warped)
