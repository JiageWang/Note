import cv2
import numpy as np


def get_sorted_bboxs(cnts):
    bboxs = [cv2.boundingRect(cnt) for cnt in cnts]
    cnts, bboxs = zip(*sorted(zip(cnts, bboxs), key=lambda x: x[1][0]))
    return cnts, bboxs


def auto_resize(img, width=None, height=None):
    if not (width or height):
        return
    if width and height:
        return
    h, w = img.shape[:2]
    if width:
        scale = width / w
        height = int(scale * h)
    else:
        scale = width / h
        width = int(scale * w)
    return cv2.resize(img, (width, height))


def cv_show(img, window=""):
    cv2.imshow(window, img)
    cv2.waitKey()


def draw_bboxs(img, bboxs):
    for i, bbox in enumerate(bboxs):
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255))
    return img


def filter_bboxs(bboxs):
    tmp = []
    for bbox in bboxs:
        x, y, w, h = bbox
        if 2.5 < w / h < 4:
            if 40 < w < 55 and 10 < h < 20:
                tmp.append(bbox)
    return tmp


def get_templates(file):
    img = cv2.imread(file, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, bboxs = get_sorted_bboxs(contours)
    templates = {}
    for i, bbox in enumerate(bboxs):
        x, y, w, h = bbox
        roi = binary[y:y + h, x:x + w]
        templates[i] = cv2.resize(roi, (55, 86))
    return templates

def match_templates():
    pass


def get_numbers_roi(file):
    img = cv2.imread(file)
    img = auto_resize(img, width=300)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernal)
    # cv_show(tophat, "tophat")

    edges = cv2.Sobel(tophat, cv2.CV_32F, 1, 0)
    edges = np.absolute(edges)
    (minVal, maxVal) = (np.min(edges), np.max(edges))
    edges = (255 * ((edges - minVal) / (maxVal - minVal)))
    edges = edges.astype("uint8")
    # cv_show(edges, "edges")

    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernal)
    # cv_show(close, "close1")

    _, binary = cv2.threshold(close, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # cv_show(binary, "binary")

    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    close = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernal)
    # cv_show(close, "close2")
    contours, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts, bboxs = get_sorted_bboxs(contours)
    bboxs = filter_bboxs(bboxs)
    rois = []

    for bbox in bboxs:
        x, y, w, h = bbox
        x1, y1, x2, y2 = x - 5, y - 5, x + w + 5, y + h + 5
        rois.append(gray[y1:y2, x1:x2])

    return img, rois, bboxs

def get_numbers_list(rois, templates):
    numbers_list = []
    for roi in rois:
        numbers = []
        # 对于每一个数字串区域
        _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_OTSU|cv2.THRESH_BINARY)
        # kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        # morph = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernal)

        # cv_show(morph, "morph")
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts, bboxs = get_sorted_bboxs(contours)
        for bbox in bboxs:
            # 对于每一个数字区域
            x, y, w, h = bbox
            num_roi = binary[y:y+h, x:x+w]
            # cv_show(num_roi, "num_roi")
            scores = []
            for value, template in templates.items():
                num_roi = cv2.resize(num_roi, (55, 86))
                result = cv2.matchTemplate(num_roi, template, cv2.TM_CCOEFF_NORMED)
                _, score, _, _ = cv2.minMaxLoc(result)
                scores.append(score)
            numbers.append(str(scores.index(max(scores))))
        numbers_list.append(numbers)
    return numbers_list






if __name__ == "__main__":
    templates = get_templates('ocr_a_reference.png')  # 模板
    img, rois, bboxs = get_numbers_roi("credit_card_01.png")  # 数字序列
    numbers_list = get_numbers_list(rois, templates)
    for bbox, numbers in zip(bboxs, numbers_list):
        x, y, w, h = bbox
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 1)
        img = cv2.putText(img, ''.join(numbers), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255) )
    print(numbers_list)
    cv_show(img, "result")



