import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours

def order_points(pts):
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def grade_omr(image_path, answer_key, num_questions, num_choices):
    image = cv2.imread(image_path)
    if image is None:
        raise Exception("Image not loaded")

    # 1. Preprocess: resize, grayscale, blur, edge detection
    image_resized = imutils.resize(image, width=700)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # 2. Find document contour (outer border of OMR sheet)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    docCnt = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break
    if docCnt is None:
        raise Exception("Could not find document boundary.")

    # 3. Perspective transform
    warped = four_point_transform(image_resized, docCnt.reshape(4,2))
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # 4. Find bubble contours
    cnts2 = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)
    questionCnts = []
    for c in cnts2:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # filter by size / aspect ratio
        if w >= 20 and h >= 20 and 0.9 <= ar <= 1.2:
            questionCnts.append(c)

    # 5. Sort questionCnts top‑to‑bottom then group into rows
    questionCnts = sorted(questionCnts, key=lambda c: cv2.boundingRect(c)[1])
    # ensure exact number: only first num_questions * num_choices
    questionCnts = questionCnts[: num_questions * num_choices]

    rows = []
    for q in range(0, len(questionCnts), num_choices):
        row = questionCnts[q : q + num_choices]
        row = sorted(row, key=lambda c: cv2.boundingRect(c)[0])
        rows.append(row)

    # 6. Detect filled bubbles
    detected = {}
    vis = warped.copy()
    for q_index, row in enumerate(rows):
        max_nonzero = -1
        filled_choice = None
        for choice_index, c in enumerate(row):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)
            if total > max_nonzero:
                max_nonzero = total
                filled_choice = choice_index
        detected[q_index] = filled_choice

        # visualize choices: draw rectangle around bubble, color green/red
        for choice_index, c in enumerate(row):
            (x, y, w, h) = cv2.boundingRect(c)
            if choice_index == filled_choice:
                if answer_key.get(q_index, None) == filled_choice:
                    color = (0, 255, 0)  # green if correct
                else:
                    color = (0, 0, 255)  # red if wrong
            else:
                color = (255, 0, 0)  # blue for not chosen
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

    # compute score
    correct = 0
    for q, ans in answer_key.items():
        if detected.get(q) == ans:
            correct += 1
    score = (correct / len(answer_key)) * 100.0

    return detected, score, vis
