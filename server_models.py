import torch
import torch.nn as nn
import cv2
from attr_models import Garbage_model, Damage_model, Fullness_model
from detector_segmentator import GraffitiDetector, CansDetector, PersonDetector, GarbageSegmentator
from align_utils import resize_mask
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'

cans_names = ['Bin', 'Can', 'Container', 'Tank', 'urn']
segmentation_names = ['garbage']
icon_width_of_image = 0.02
min_icon_witdth = 16

name_color = {
     'ad': (0, 0, 255), #Blue
     'person': (0, 0, 0), #Black
     'graffiti': (0, 255, 0), #Lime
     'garbage': (0, 128, 128), #Teal
     'Ad': (192, 192, 192), #Silver
     'Place': (255, 0, 255), #Fuchsia
     'Bin': (255, 0, 0), #Red
     'Can': (128, 0, 0), #Maroon
     'Container': (255, 255, 0), #Yellow
     'Tank': (128, 128, 0), #Olive
     'urn': (0, 128, 0),#Green
}
   
attr_icon = {
    'ok': cv2.imread('./icons/ok.png'),
    'flipped': cv2.imread('./icons/flipped.png'),
    'broken': cv2.imread('./icons/broken.png'),
    'empty': cv2.imread('./icons/empty.png'),
    'half': cv2.imread('./icons/half.png'),
    'full': cv2.imread('./icons/full.png'),
    'bulk': cv2.imread('./icons/bulk.png'),
    'TKO': cv2.imread('./icons/tko.png'),
    'KGM': cv2.imread('./icons/kgm.png'),
}

garbage_model = Garbage_model(device)
damage_model = Damage_model(device)
fullness_model = Fullness_model(device)
graffiti = GraffitiDetector(device, fp16=False)
cans = CansDetector(device, fp16=False)
people = PersonDetector(device, fp16=False)
garbage = GarbageSegmentator(device, fp16=False)

def draw_icon(image, box, icon, number_of_icon=1):
    rectangle_coords = box
    top_right_x = max(rectangle_coords[0], rectangle_coords[2])
    top_right_y = min(rectangle_coords[1], rectangle_coords[3])
    resize_per = int(max(max(image.shape[:2]) * icon_width_of_image, min_icon_witdth))
    resized_icon = cv2.resize(icon, (resize_per, resize_per), interpolation = cv2.INTER_AREA)
    icon_height, icon_width = resized_icon.shape[:2]
    icon_top_left_x = top_right_x - icon_width * number_of_icon
    icon_top_left_y = top_right_y - icon_height
    if icon_top_left_y >= 0 and icon_top_left_x >= 0:
        image[int(icon_top_left_y):int(icon_top_left_y + icon_height), 
              int(icon_top_left_x):int(icon_top_left_x + icon_width)] = resized_icon
    return image

def drawRectangle(image_cv2, box_points, color, width=4):
    cv2.rectangle(image_cv2, (box_points[0], box_points[1]), (box_points[2], box_points[3]), color, width)

def drawMask(image_cv2, mask, color, alpha=0.5, resize=None):
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image_cv2, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    if resize is not None:
        image_cv2 = cv2.resize(image_cv2.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)
    return cv2.addWeighted(image_cv2, 1 - alpha, image_overlay, alpha, 0)
    
def draw_detection_results(image_cv2, res, key):
    box_points = res['box_points']
    color = name_color[key]
    if key in segmentation_names:
        mask = resize_mask(image_cv2, res['mask'])
        image_cv2 = drawMask(image_cv2, mask, color)
    number_of_icon = 0
    for dict_key in res:
        if dict_key not in ['box_points', 'mask']:
            number_of_icon = number_of_icon + 1
            attr = res[dict_key]
            icon = attr_icon[attr]
            image_cv2 = draw_icon(image_cv2, box_points, icon, number_of_icon)
            
    drawRectangle(image_cv2, box_points, color)
    return image_cv2