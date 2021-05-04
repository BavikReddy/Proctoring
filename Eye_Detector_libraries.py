import cv2
import numpy as np
import tensorflow as tf



def eye_on_mask(mask, side, shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1] + points[2][1]) // 2
    r = points[3][0]
    b = (points[4][1] + points[5][1]) // 2
    return mask, [l, t, r, b]

def find_eyeball_position(end_points, cx, cy):
    """Find and return the eyeball positions, i.e. left or right or top or normal"""
    x_ratio = (end_points[0] - cx)/(cx - end_points[2])
    y_ratio = (cy - end_points[1])/(end_points[3] - cy)
    #print('x_ratio-',x_ratio)
    #print('y_ratio-',y_ratio)
    if x_ratio > 1.5:
        return 1
    elif x_ratio < 0.85:
        return 2
    elif y_ratio < 0.7:
        return 3
    else:
        return 0

def contouring(thresh, mid, img, end_points, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        #print('cx-',cx)
        #print('cy-',cy)
        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        pass


def process_thresh(thresh):

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 3)
    thresh = cv2.bitwise_not(thresh)
    return thresh

def print_eye_pos(img, left, right):

    #if left == right and left != 0:
    text = ''
    if left == 1:
        #print('Looking left')
        text = 'Looking left'
    elif left == 2:
        #print('Looking right')
        text = 'Looking right'
    elif left == 3:
        #print('Looking up')
        text = 'Looking up'
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (30, 30), font,
                       1, (0, 255, 255), 2, cv2.LINE_AA)
    return text

def nothing(x):
    pass

