import re
import xml.etree.ElementTree as et
import time
import os
import random
from xmlrpc.client import Error
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import torch
#print(torch.cuda.is_available())
from save_results import save_results
import pytesseract
import imutils

# change to match your tesseract installation path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# config
data_dir = 'data/photos/'
annots_file = 'data/annotations.xml'
test_count = 100
seed = 69


# read xml
def parse_annots(path):
    tree = et.parse(path)
    root = tree.getroot()
    annots = {}
    for img in root.findall('.//image'):
        name = img.get('name')
        boxes = []
        for box in img.findall('box'):
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            text = box.find("attribute").text
            boxes.append({'bbox': [xtl, ytl, xbr, ybr], 'text': text})
        annots[name] = boxes
    return annots


# intersect. over union
def iou(box_a, box_b):
    # inters. corners
    xtl = max(box_a[0], box_b[0])
    ytl = max(box_a[1], box_b[1])
    xbr = min(box_a[2], box_b[2])
    ybr = min(box_a[3], box_b[3])
    inter_w = max(0, xbr - xtl)
    inter_h = max(0, ybr - ytl)
    inter_area = inter_w * inter_h
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = float(box_a_area + box_b_area - inter_area)
    return inter_area / union if union > 0 else 0


# loading the model. Use GPU if possible
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = YOLO('models/license-plate-finetune-v1x.pt') # put your model here
model.to(device)

# polish number plates use latin chars - no 'pl' lang needed
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())


def detect(img):
    results = model(img)[0]
    # return if no detection
    if len(results.boxes) == 0:
        return None, None
    boxes = results.boxes.xyxy.cpu().numpy()
    # confidence values of the detections
    confs = results.boxes.conf.cpu().numpy()
    # best confidence selection
    idx = np.argmax(confs)
    x1, y1, x2, y2 = boxes[idx]
    bbox = [float(x1), float(y1), float(x2), float(y2)]
    crop = img[int(y1):int(y2), int(x1):int(x2)]
    return bbox, crop


def preprocessing(crop):
    scale = 1.5
    res = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # trying to improve accuracy with various filters
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    #equ = cv2.equalizeHist(gray)
    blur = cv2.bilateralFilter(gray, 11, 30, 23)
    thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_OTSU)[1]
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    morphed2 = cv2.morphologyEx(morphed, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

    # contour detection and cropping. This vastly improves accuracy
    keypoints = cv2.findContours(morphed2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    best_contour, best_contour_area = None, 0
    # biggest contour
    for contour in contours:
        area = cv2.contourArea(contour)
        if best_contour_area < area:
            best_contour = contour
            best_contour_area = area
            continue

    x, y, w, h = cv2.boundingRect(best_contour)

    return morphed2[y:y + h, x:x + w]


# ocr
def ocr(crop):
    if crop is None or crop.size == 0:
        return ''

    crop_2 = preprocessing(crop)

    config = '--oem 1 --psm 7 -c tessedit_char_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"'
    text = pytesseract.image_to_string(crop_2, lang='eng', config=config)

    # postprocessing - licence plates have max 8 chars, no spaces and are capitalised
    text = text.upper().replace(' ', '').replace('\n', '')[:8]

    return text


def evaluate(annots, n=100):
    all_files = [f for f in annots.keys() if f.endswith('.jpg')]
    # test on imgs: 61.jpg - 194, because 1-60 were used to create tesseract lang
    filtered = []
    for f in all_files:
        base = os.path.splitext(f)[0]
        m = re.match(r"^(\d+)", base)
        if not m:
            continue
        num = int(m.group(1))
        if 61 <= num <= 194:
            filtered.append(f)
    if len(filtered) < n:
        n = len(filtered)
        print(f"You goofus! Not enough images for testing! I'll process only {n} imgs!")
    if n == 0:
        raise Error("You are very unwise! Cannot evaluate 0 photos.")

    test_imgs = random.sample(filtered, n)
    correct = 0
    ious = []
    detects = {} # dict of detected texts

    start = time.time()
    for file_name in test_imgs:
        img_path = os.path.join(data_dir, file_name)
        img = cv2.imread(img_path)

        info = annots[file_name][0] # bbox (coords) and text from the plate
        det = detect(img) # bbox & cropped img

        # continue if nothing detected
        if det[0] is None:
            detects[file_name] = ''
            ious.append(0)
            continue

        bbox, crop = det
        iou_val = iou(info['bbox'], bbox) # calculate iou
        ious.append(iou_val)

        detected_text = ocr(crop)
        detects[file_name] = detected_text
        # check if reading was correct
        if detected_text == info['text'].strip().upper():
            correct += 1

    t = time.time() - start
    accur = (correct / n) * 100
    if ious:
        avarage_iou = np.mean(ious)
    else:
        avarage_iou = 0

    return accur, t, avarage_iou, detects


if __name__ == '__main__':
    random.seed(seed)
    annotations = parse_annots(annots_file) # contains file names, box coords and plates' text

    # detection & ocr
    acc, run_time, avg_iou, detections = evaluate(annotations, test_count)

    # results
    print(f"\nAcc: {acc:.2f}%, avg IoU: {avg_iou:.3f}, t: {run_time:.2f}s")
    save_results(annotations, detections)
