import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model once
model = YOLO("yolov8n.pt")

def crop_breast_region(image):

    img = image.copy()
    h, w, _ = img.shape

    # Detect person
    results = model(img)
    boxes = results[0].boxes.xyxy

    if len(boxes) == 0:
        return img, img, img

    x1,y1,x2,y2 = boxes[0].cpu().numpy().astype(int)

    # Chest area
    top = int(y1 + (y2-y1)*0.35)
    bottom = int(y1 + (y2-y1)*0.75)

    left = int(x1 + (x2-x1)*0.05)
    right = int(x2 - (x2-x1)*0.05)

    crop = img[top:bottom, left:right]

    return crop


def extract_breast_curves(crop):

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)

    edges = cv2.Canny(blur,50,150)

    kernel = np.ones((5,5),np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return crop, crop, crop

    largest = max(contours, key=cv2.contourArea)

    epsilon = 0.003 * cv2.arcLength(largest, True)
    smooth = cv2.approxPolyDP(largest, epsilon, True)

    curve_img = crop.copy()
    cv2.drawContours(curve_img,[smooth],-1,(255,255,0),2)

    h,w,_ = curve_img.shape
    center_x = w//2
    cv2.line(curve_img,(center_x,0),(center_x,h),(255,0,0),2)

    left_breast = curve_img[:, :center_x]
    right_breast = curve_img[:, center_x:]

    return curve_img, left_breast, right_breast