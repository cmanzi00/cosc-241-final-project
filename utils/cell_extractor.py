from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2


def extract_cell(cell):
    # Apply threshholding and border clearance
    thresh = cv2.threshold(cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    # processed = cv2.Laplacian(thresh, cv2.CV_64F)
    # processed = cv2.convertScaleAbs(processed)

    # Find contours in the thresholded cell
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Consider empty cell if no contours found
    if len(cnts) == 0:
        return None

    # Otherwise, find largest countour in cell and create mask
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype=np.uint8)
    cv2.drawContours(mask, [c], -1, 255, -1)

    # Compute the percentage of masked pixels relative to the image pixels
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    # Check if cell is empty
    if percentFilled < 0.03:
        return None

    # Apply mask to the cell
    cell_value = cv2.bitwise_and(thresh, thresh, mask=mask)

    return cell_value
