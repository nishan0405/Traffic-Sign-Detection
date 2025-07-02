import os
import cv2
import numpy as np
import math
from classification import training, getLabel

SIGNS = ["ERROR", "STOP", "TURN LEFT", "TURN RIGHT", "DO NOT TURN LEFT", "DO NOT TURN RIGHT", "ONE WAY", "SPEED LIMIT", "OTHER"]

def clean_images():
    file_list = os.listdir('./')
    for file_name in file_list:
        if '.png' in file_name:
            os.remove(file_name)

def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(img_hist_equalized))
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)
    gray = cv2.cvtColor(LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian(gray, cv2.CV_8U, 3, 3, 2)
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image

def binarization(image):
    thresh = cv2.threshold(image, 32, 255, cv2.THRESH_BINARY)[1]
    return thresh

def preprocess_image(image):
    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    return image

def removeSmallComponents(image, threshold):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    img2 = np.zeros((output.shape), dtype=np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def findContour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [np.reshape(contour, (-1, 1, 2)).astype(np.int32) for contour in contours]
    return contours

def contourIsSign(perimeter, centroid, threshold):
    result = []
    for p in perimeter:
        p = p[0]
        distance = math.sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result]
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold:
        return True, max_value + 2
    else:
        return False, max_value + 2

def cropSign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height - 1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width - 1])
    return image[top:bottom, left:right]

def findLargestSign(image, contours, threshold, distance_threshold):
    max_distance = 0
    coordinate = None
    sign = None
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contourIsSign(c, [cX, cY], 1 - threshold)
        if is_sign and distance > max_distance and distance > distance_threshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1, 2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis=0)
            coordinate = [(left - 2, top - 2), (right + 3, bottom + 1)]
            sign = cropSign(image, coordinate)
    return sign, coordinate

def localization(image, min_size_components, similitary_contour_with_circle, model, count, current_sign_type):
    original_image = image.copy()
    binary_image = preprocess_image(image)
    binary_image = removeSmallComponents(binary_image, min_size_components)
    binary_image = cv2.bitwise_and(binary_image, binary_image, mask=remove_other_color(image))
    contours = findContour(binary_image)
    sign, coordinate = findLargestSign(original_image, contours, similitary_contour_with_circle, 15)
    text = ""
    sign_type = -1
    if sign is not None:
        sign_type = getLabel(model, sign)
        sign_type = sign_type if sign_type <= 8 else 8
        text = SIGNS[sign_type]
        cv2.imwrite(str(count) + '_' + text + '.png', sign)
    if sign_type > 0 and sign_type != current_sign_type:
        cv2.rectangle(original_image, coordinate[0], coordinate[1], (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(original_image, text, (coordinate[0][0], coordinate[0][1] - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return coordinate, original_image, sign_type, text

def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 128, 0])
    upper_blue = np.array([215, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    lower_white = np.array([0, 0, 128], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([170, 150, 50], dtype=np.uint8)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask = cv2.bitwise_or(mask_1, mask_black)
    return mask

def main():
    clean_images()
    model = training()
    cap = cv2.VideoCapture(0)
    similitary_contour_with_circle = 0.65
    min_size_components = 300
    current_sign = None
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        coordinate, image, sign_type, text = localization(frame, min_size_components, similitary_contour_with_circle, model, count, current_sign)
        if coordinate is not None:
            cv2.rectangle(image, coordinate[0], coordinate[1], (255, 255, 255), 1)
        cv2.imshow('Traffic Sign Recognition', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
import os
import cv2
import numpy as np
import math
from classification import training, getLabel

SIGNS = ["ERROR", "STOP", "TURN LEFT", "TURN RIGHT", "DO NOT TURN LEFT", "DO NOT TURN RIGHT", "ONE WAY", "SPEED LIMIT", "OTHER"]

def clean_images():
    file_list = os.listdir('./')
    for file_name in file_list:
        if '.png' in file_name:
            os.remove(file_name)

def constrastLimit(image):
    img_hist_equalized = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    channels = list(cv2.split(img_hist_equalized))
    channels[0] = cv2.equalizeHist(channels[0])
    img_hist_equalized = cv2.merge(channels)
    img_hist_equalized = cv2.cvtColor(img_hist_equalized, cv2.COLOR_YCrCb2BGR)
    return img_hist_equalized

def LaplacianOfGaussian(image):
    LoG_image = cv2.GaussianBlur(image, (3,3), 0)
    gray = cv2.cvtColor(LoG_image, cv2.COLOR_BGR2GRAY)
    LoG_image = cv2.Laplacian(gray, cv2.CV_8U, 3, 3, 2)
    LoG_image = cv2.convertScaleAbs(LoG_image)
    return LoG_image

def binarization(image):
    thresh = cv2.threshold(image, 32, 255, cv2.THRESH_BINARY)[1]
    return thresh

def preprocess_image(image):
    image = constrastLimit(image)
    image = LaplacianOfGaussian(image)
    image = binarization(image)
    return image

def removeSmallComponents(image, threshold):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -1]; nb_components = nb_components - 1
    img2 = np.zeros((output.shape), dtype=np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2

def findContour(image):
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [np.reshape(contour, (-1, 1, 2)).astype(np.int32) for contour in contours]
    return contours

def contourIsSign(perimeter, centroid, threshold):
    result = []
    for p in perimeter:
        p = p[0]
        distance = math.sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result]
    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold:
        return True, max_value + 2
    else:
        return False, max_value + 2

def cropSign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height - 1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width - 1])
    return image[top:bottom, left:right]

def findLargestSign(image, contours, threshold, distance_threshold):
    max_distance = 0
    coordinate = None
    sign = None
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        is_sign, distance = contourIsSign(c, [cX, cY], 1 - threshold)
        if is_sign and distance > max_distance and distance > distance_threshold:
            max_distance = distance
            coordinate = np.reshape(c, [-1, 2])
            left, top = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis=0)
            coordinate = [(left - 2, top - 2), (right + 3, bottom + 1)]
            sign = cropSign(image, coordinate)
    return sign, coordinate

def localization(image, min_size_components, similitary_contour_with_circle, model, count, current_sign_type):
    original_image = image.copy()
    binary_image = preprocess_image(image)
    binary_image = removeSmallComponents(binary_image, min_size_components)
    binary_image = cv2.bitwise_and(binary_image, binary_image, mask=remove_other_color(image))
    contours = findContour(binary_image)
    sign, coordinate = findLargestSign(original_image, contours, similitary_contour_with_circle, 15)
    text = ""
    sign_type = -1
    if sign is not None:
        sign_type = getLabel(model, sign)
        sign_type = sign_type if sign_type <= 8 else 8
        text = SIGNS[sign_type]
        cv2.imwrite(str(count) + '_' + text + '.png', sign)
    if sign_type > 0 and sign_type != current_sign_type:
        cv2.rectangle(original_image, coordinate[0], coordinate[1], (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(original_image, text, (coordinate[0][0], coordinate[0][1] - 10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    return coordinate, original_image, sign_type, text

def remove_other_color(img):
    frame = cv2.GaussianBlur(img, (3, 3), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 128, 0])
    upper_blue = np.array([215, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    lower_white = np.array([0, 0, 128], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    lower_black = np.array([0, 0, 0], dtype=np.uint8)
    upper_black = np.array([170, 150, 50], dtype=np.uint8)
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_1 = cv2.bitwise_or(mask_blue, mask_white)
    mask = cv2.bitwise_or(mask_1, mask_black)
    return mask

def main():
    clean_images()
    model = training()
    cap = cv2.VideoCapture(0)
    similitary_contour_with_circle = 0.65
    min_size_components = 300
    current_sign = None
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        coordinate, image, sign_type, text = localization(frame, min_size_components, similitary_contour_with_circle, model, count, current_sign)
        if coordinate is not None:
            cv2.rectangle(image, coordinate[0], coordinate[1], (255, 255, 255), 1)
        cv2.imshow('Traffic Sign Recognition', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()