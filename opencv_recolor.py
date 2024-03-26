import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def readImage(img_name):
    img = cv2.imread(img_name)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def resizeAndPad(img):
    height, width = img.shape[:2]
    scaled_height, scaled_width = height+2, width+2
    interpolate = cv2.INTER_CUBIC
    aspect = width/height 
    if aspect > 1:
        new_width = scaled_width
        new_height = np.round(new_width/aspect).astype(int)
        pad_vertical = (scaled_height-new_height)/2
        pad_top, pad_bottom = np.floor(pad_vertical).astype(int), np.ceil(pad_vertical).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1:
        new_height = scaled_height
        new_width = np.round(new_height*aspect).astype(int)
        pad_horizontal = (scaled_width-new_width)/2
        pad_left, pad_right = np.floor(pad_horizontal).astype(int), np.ceil(pad_horizontal).astype(int)
        pad_top, pad_bottom = 0, 0
    else:
        new_height, new_width = scaled_height, scaled_width
        pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
    pad_color = [255]*3
    scaled_img = cv2.resize(img, (new_width, new_height), interpolation=interpolate)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=pad_color)
    return scaled_img

def getOutlineImg(img):
    return cv2.Canny(img,50,200)

def getColoredImage(img, new_color):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_img)
    hsv_color = cv2.cvtColor(np.uint8([[new_color]]), cv2.COLOR_RGB2HSV)
    h.fill(hsv_color[0][0][0])  
    s.fill(hsv_color[0][0][1])
    new_hsv_img = cv2.merge([h, s, v])
    new_rgb_img = cv2.cvtColor(new_hsv_img, cv2.COLOR_HSV2RGB)
    return new_rgb_img

def selectWall(outline_img, position):
    wall = outline_img.copy()
    scaled_mask = resizeAndPad(outline_img)
    cv2.floodFill(wall, scaled_mask, position, 255)  
    cv2.subtract(wall, outline_img, wall) 
    return wall

def mergeImages(img, colored_img, wall):
    colored_img = cv2.bitwise_and(colored_img, colored_img, mask=wall)
    marked_img = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(wall))
    final_img = cv2.bitwise_xor(colored_img, marked_img)
    return final_img

def saveImage(img_name, img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_name= img_name[10:]
    cv2.imwrite("./tester/" + img_name, img)

def changeColor(img_name, position, new_color):
    img = readImage(img_name)
    colored_img = getColoredImage(img, new_color)
    outline_img = getOutlineImg(img)
    selected_wall = selectWall(outline_img, position)
    final_img = mergeImages(img, colored_img, selected_wall)
    saveImage(img_name, final_img)

images = glob.glob('Data/Test/33012f501af95eecd41803710646e57e87c2680d.jpg')

for img in images: 
    changeColor(img, (10,20), [111, 209, 201])
