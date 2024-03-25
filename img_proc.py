import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def readImage(img_name):
    img = cv2.imread(img_name)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resizeAndPad(img, size, pad_color=0):
    h, w = img.shape[:2]
    sh, sw = size
    if h > sh or w > sw:
        interp = cv2.INTER_AREA
    else: 
        interp = cv2.INTER_CUBIC
    aspect = w/h  
    if aspect > 1:
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else:
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0
    if len(img.shape) == 3 and not isinstance(pad_color, (list, tuple, np.ndarray)): # color image but only one color provided
        pad_color = [pad_color]*3
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return scaled_img

def getOutlineImg(img):
    return cv2.Canny(img,50,200)

def getColoredImage(img, new_color):
    hsv_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)
    new_hsv_image = hsv_image
    color = np.uint8([[new_color]])
    hsv_color = cv2.cvtColor(color, cv2.COLOR_RGB2HSV)
    h.fill(hsv_color[0][0][0])  
    s.fill(hsv_color[0][0][1])
    new_hsv_image = cv2.merge([h, s, v])
    new_rgb_image = cv2.cvtColor(new_hsv_image, cv2.COLOR_HSV2RGB)
    return new_rgb_image

def selectWall(outline_img, position):
    h, w = outline_img.shape[:2]
    wall = outline_img.copy()
    scaled_mask = resizeAndPad(outline_img, (h+2,w+2), 255)
    cv2.floodFill(wall, scaled_mask, position, 255)  
    cv2.subtract(wall, outline_img, wall) 
    return wall

def mergeImages(img, colored_image, wall):
    colored_image = cv2.bitwise_and(colored_image, colored_image, mask=wall)
    marked_img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(wall))
    final_img = cv2.bitwise_xor(colored_image, marked_img)
    return final_img

def saveImage(img_name, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_name= img_name[10:]
    cv2.imwrite("./CV2Edited/" + img_name, img)

def changeColor(image_name, position, new_color):
    img = readImage(image_name)
    colored_image = getColoredImage(img, new_color)
    outline_img = getOutlineImg(img)
    selected_wall = selectWall(outline_img, position)
    final_img = mergeImages(img, colored_image, selected_wall)
    saveImage(image_name, final_img)

images = glob.glob('Data/Test/*.jpg')

for img in images: 
    changeColor(img, (10,20), [111, 209, 201])
