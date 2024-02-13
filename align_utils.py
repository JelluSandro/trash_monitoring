import math

import PIL.Image
import cv2
import numpy as np
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image

cans_resize = 0.35
resize_rectangle_x = 0.2
resize_percentage = 0.35
resize_rectangle_y = 0.5
default_width = 640

def crop_img(img_pil, box_points):
    croped_img = img_pil.crop(box_points)
    return croped_img
    
def coords_outs(x, max_x):
    return int(max(min(x, max_x), 0))
    
def count_max_h_w(xmin, ymin, xmax, ymax):
    h = ymax - ymin
    w = xmax - xmin
    return max(w, h) / 2
    
def get_h_w(img_pil_or_cv2):
    if isinstance(img_pil_or_cv2, PIL.Image.Image):
        img_h, img_w = img_pil_or_cv2.size
    elif isinstance(img_pil_or_cv2, np.ndarray):
        img_w, img_h = img_pil_or_cv2.shape[:2]
    else:
        raise RuntimeError(f"Unknown image type: {type(img_pil_or_cv2)}. "
                           f"Expected torch.Tensor or PIL.Image.Image")
    return img_h, img_w
    
def change_coords_for_cans(img_pil, box_points):
    img_h, img_w = get_h_w(img_pil)
    xmin, ymin, xmax, ymax = box_points
    max_h_w = count_max_h_w(xmin, ymin, xmax, ymax)
    xmin = coords_outs(xmin, img_h)
    xmax = coords_outs(xmax, img_h)
    ymin = coords_outs(ymin - max_h_w * cans_resize, img_w)
    ymax = coords_outs(ymax, img_w)
    return (xmin, ymin, xmax, ymax)
    
def crop_cans(img_pil, box_points):
    return crop_img(img_pil, change_coords_for_cans(img_pil, box_points))

def resize_mask(image, mask):
    img_h, img_w = get_h_w(image)
    if mask.shape[:2] == (default_width, default_width):
        if img_h < img_w:
            per = default_width / img_w
            res_side = per * img_h
            padding = (default_width - res_side) / 2
            mask = mask[0:default_width, int(padding):int(padding + res_side)]
        elif img_w < img_h:
            per = default_width / img_h
            res_side = per * img_w
            padding = (default_width - res_side) / 2
            mask = mask[int(padding):int(padding + res_side), 0:default_width]
    mask_resize = cv2.resize(mask, (img_h, img_w))
    return mask_resize
    
def crop_segmentation(img_pil, box_points, mask):
    mask_resize = resize_mask(img_pil, mask)
    xmin, ymin, xmax, ymax = box_points
    image_cropped = img_pil.crop(box_points)
    image_cv = cv2.cvtColor(np.array(image_cropped), cv2.COLOR_RGB2BGR)
    crop_mask = mask_resize[ymin:ymax, xmin:xmax]
    mask_cv = cv2.cvtColor(crop_mask, cv2.COLOR_RGB2BGR)
    result = cv2.bitwise_and(image_cv, mask_cv.astype(np.uint8) * 255)
    result_image = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    return result_image
    
def draw_rectangles_on_cans(img_pil_or_cv2, box_points):
    img_h, img_w = get_h_w(img_pil_or_cv2)
    (xmin, ymin, xmax, ymax) = change_coords_for_cans(img_pil_or_cv2, box_points)
    max_h_w = count_max_h_w(xmin, ymin, xmax, ymax)
    xmin_rectangle = coords_outs(xmin - max_h_w * resize_rectangle_x, img_h)
    xmax_rectangle = coords_outs(xmax + max_h_w * resize_rectangle_x, img_h)
    ymin_rectangle = coords_outs(ymin - max_h_w * resize_rectangle_y, img_w)
    ymax_rectangle = coords_outs(ymax - max_h_w * resize_rectangle_y, img_w)
    if isinstance(img_pil_or_cv2, PIL.Image.Image):
        img_pil_or_cv2.paste(
            Image.new("RGB", 
            (xmax_rectangle - xmin_rectangle, ymax_rectangle - ymin_rectangle),
            (255, 255, 255)),
            (xmin_rectangle, ymin_rectangle))
    else:
        white_rectangle = np.full(
        ((ymax_rectangle - ymin_rectangle), (xmax_rectangle - xmin_rectangle), 3),
        255, dtype=np.uint8)
        img_pil_or_cv2[
            ymin_rectangle:ymax_rectangle,
            xmin_rectangle:xmax_rectangle
        ] = white_rectangle
    return img_pil_or_cv2
