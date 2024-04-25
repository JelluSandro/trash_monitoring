from io import BytesIO
import numpy as np

from tqdm import tqdm
from server_models import cans, people, garbage, graffiti, garbage_model, fullness_model, damage_model, cans_names, segmentation_names
from align_utils import draw_rectangles_on_cans
from align_utils import crop_img, crop_cans, crop_segmentation
from server_utils import normilise_batch_cv2
from PIL import Image
from enum import Enum
import math

import cv2

def get_crop_detection_segmentation(batch, batch_detections, save=False, crop_only_attr=True):
    assert len(batch) == len(batch_detections)
    
    for (image, dict_of_cls) in tqdm(zip(batch, batch_detections)):
        pil_img = Image.fromarray(image[:, :, ::-1])
        for key in dict_of_cls:
            for res in dict_of_cls[key]:
                box_points = res['box_points']
                out_img = None
                if key in cans_names:
                    out_img = crop_cans(pil_img, box_points)
                elif key in segmentation_names:
                    mask = res['mask']
                    if mask is not None and mask.size != 0:
                        out_img = crop_segmentation(pil_img, box_points, mask)
                elif not crop_only_attr:
                    out_img = crop_img(pil_img, box_points)
                with BytesIO() as output:
                    if save and out_img is not None :
                        out_img.save(output, format="PNG")
                    yield output.getvalue()
                    
def add_attributes_in_batch(batch, model, batch_res, name):
    if batch != []:
        batch_normilise = normilise_batch_cv2(batch)
        attr = model.run(batch_normilise)
        for ind, res in enumerate(batch_res):
            res[name] = attr[ind]

def add_attributes(batch, batch_detections, garbageAttribute, fullnesAttribute, damageAttribute, save=False):
    assert len(batch) == len(batch_detections)
    
    croped_batch_cans = []
    croped_batch_cans_res = []
    croped_batch_seg = []
    croped_batch_seg_res = []
    for (image, dict_of_cls) in tqdm(zip(batch, batch_detections)):
        pil_img = Image.fromarray(image[:, :, ::-1])
        for key in dict_of_cls:
            for res in dict_of_cls[key]:
                box_points = res['box_points']
                if (fullnesAttribute or damageAttribute) and (key in cans_names):
                    croped_batch_cans.append(crop_cans(pil_img, box_points))
                    croped_batch_cans_res.append(res)
                elif key in segmentation_names:
                    mask = res['mask']
                    if not save:
                        del res['mask']
                    if mask is not None and mask.size != 0:
                        croped_batch_seg.append(crop_segmentation(pil_img, box_points, mask))
                        croped_batch_seg_res.append(res)
    if damageAttribute:
        add_attributes_in_batch(croped_batch_cans, damage_model, croped_batch_cans_res, 'damage')
    if fullnesAttribute:
        add_attributes_in_batch(croped_batch_cans, fullness_model, croped_batch_cans_res, 'fullness')
    if garbageAttribute:
        add_attributes_in_batch(croped_batch_seg, garbage_model, croped_batch_seg_res, 'dirtiness')
    
class EmptyResualt(Enum):
    ALL = 1
    NOTHING = 0
    
def update_result_data(result_data, detection_data):
    for ind, dict_objs in enumerate(result_data):
        dict_objs.update(detection_data[ind])
            
def get_compose_mask(result_data):
    composed_mask = result_data[0]['mask'] if len(result_data) > 0 else EmptyResualt.NOTHING
    for res in result_data[1:]:
        composed_mask = cv2.bitwise_or(composed_mask, res['mask'])
    return composed_mask
    
def my_bitwise_and(mask1, mask2):
    if type(mask1) == EmptyResualt:
        if mask1 == EmptyResualt.ALL:
            return mask2
        if mask1 == EmptyResualt.NOTHING:
            return EmptyResualt.NOTHING
    if type(mask2) == EmptyResualt:
        if mask2 == EmptyResualt.ALL:
            return mask1
        if mask2 == EmptyResualt.NOTHING:
            return EmptyResualt.NOTHING
    return cv2.bitwise_and(mask1, mask2)
    
def my_bitwise_or(mask1, mask2):
    if type(mask1) == EmptyResualt:
        if mask1 == EmptyResualt.ALL:
            return EmptyResualt.ALL
        if mask1 == EmptyResualt.NOTHING:
            return mask2
    if type(mask2) == EmptyResualt:
        if mask2 == EmptyResualt.ALL:
            return EmptyResualt.ALL
        if mask2 == EmptyResualt.NOTHING:
            return mask1
    return cv2.bitwise_or(mask1, mask2)
    
def my_bitwise_not(mask):
    if type(mask) == EmptyResualt:
        if mask == EmptyResualt.ALL:
            return EmptyResualt.NOTHING
        if mask == EmptyResualt.NOTHING:
            return EmptyResualt.ALL
    return cv2.bitwise_not(mask)

def get_difference(masks, mask_shift):
    compose_masks = []
    for ind in range(len(masks)):
        compose_masks.append(my_bitwise_or(masks[ind], mask_shift[ind]))
    difference = my_bitwise_not(compose_masks[0])
    size = len(masks)
    for ind in range(1, size):
        mask_to_compose = my_bitwise_not(compose_masks[ind]) if ind < size / 2 else compose_masks[ind]
        difference = my_bitwise_and(difference, mask_to_compose)
    return difference
    
def get_difference_garbage_mask(image1, image2, image3, image4):
    images = [image1, image2, image3, image4]
    out_data = get_all_detection_by_batch(images, False, True, False, True, save_mask=True)
    composed_masks = []
    for res in out_data:
        garbage_predict = res['garbage']
        composed_mask = get_compose_mask(garbage_predict)
        composed_masks.append(composed_mask)
    return get_difference(*composed_masks)
    
def dfs(matrix, x, y, visited, num):
    if x < 0 or x >= len(matrix) or y < 0 or y >= len(matrix[0]) or matrix[x][y] == 0.0 or visited[x][y] != -1:
        return 0
    visited[x][y] = num
    size = 1
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nsize = dfs(matrix, x + dx, y + dy, visited, num)
        size += nsize
    return size
    
def is_new_garbage(s, a, b):
    # сравниваем с 75%ми площади эллипса 
    s_ellipse = math.pi * (a / 2) * (b / 2) * 0.75
    return (s > s_ellipse) and (s > 200)

def find_figures(matrix):
    visited = [[-1 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    figures = []
    vertical_len = [0 for _ in range(len(matrix[0]))]
    for i in range(len(matrix)):
        horizontal_len = 0
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1.0:
                horizontal_len += 1
                vertical_len[j] += 1
                if visited[i][j] == -1:
                    size = dfs(matrix, i, j, visited, len(figures))
                    figures.append({
                        'size': size,
                        'max_row': 1,
                        'max_col': 1
                    })
                fig = figures[visited[i][j]]
                fig['max_row'] = max(horizontal_len, fig['max_row'])
                fig['max_col'] = max(vertical_len[j], fig['max_col'])
            else:
                horizontal_len = 0
                vertical_len[j] = 0

    for fi in figures:
        if is_new_garbage(fi['size'], fi['max_row'], fi['max_col']):
            return True
    return False
    
def get_all_detection_by_batch(batch, detectPeople, detectCans, detectGraffiti, detectGarbage, save_mask=False):
    result_data = [{} for _ in batch]
    detection_functions = {
        'Graffiti': (detectGraffiti, graffiti.run),
        'Cans': (detectCans, cans.run),
        'People': (detectPeople, people.run),
        'Garbage': (detectGarbage, garbage.run)
    }
    for detection_name, (detectCondition, detection_func) in detection_functions.items():
        if detectCondition:
            if detection_name != 'Garbage':
                detection_data = detection_func(batch)
                if detection_name == 'Cans':
                    edited_batch = []
                    for (image, info) in tqdm(zip(batch, detection_data)):
                        edited_image = image.copy()
                        for key in cans_names:
                            for res in info[key]:
                                box_points = res['box_points']
                                draw_rectangles_on_cans(edited_image, box_points)
                        edited_batch.append(edited_image)
            else:
                detection_data = detection_func(batch if not detectCans else edited_batch, save_mask)
            update_result_data(result_data, detection_data)

    return result_data