from io import BytesIO
import numpy as np

from tqdm import tqdm
from server_models import cans, people, garbage, graffiti, garbage_model, fullness_model, damage_model, cans_names, segmentation_names
from align_utils import draw_rectangles_on_cans
from align_utils import crop_img, crop_cans, crop_segmentation
from server_utils import normilise_batch_cv2
from PIL import Image

def get_crop_detection_segmentation(batch, batch_detections, save=False, crop_only_attr=True):
    assert len(batch) == len(batch_detections)
    
    for (image, dict_of_cls) in tqdm(zip(batch, batch_detections)):
        pil_img = Image.fromarray(image[:, :, ::-1])
        for key in dict_of_cls:
            for res in dict_of_cls[key]:
                box_points = res['box_points']
                if key in cans_names:
                    out_img = crop_cans(pil_img, box_points)
                elif key in segmentation_names:
                    mask = res['mask']
                    if mask is not None and mask.size != 0:
                        out_img = crop_segmentation(pil_img, box_points, mask)
                elif not crop_only_attr:
                    out_img = crop_img(pil_img, box_points)
                with BytesIO() as output:
                    if save:
                        out_img.save(output, format="PNG")
                    yield output.getvalue()
                    
def add_attributes_in_batch(batch, model, batch_res, name):
    if batch != []:
        batch_normilise = normilise_batch_cv2(batch)
        attr = model.run(batch_normilise)
        for ind, res in enumerate(batch_res):
            res[name] = attr[ind]

def add_attributes(batch, batch_detections, save=False):
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
                if key in cans_names:
                    croped_batch_cans.append(crop_cans(pil_img, box_points))
                    croped_batch_cans_res.append(res)
                elif key in segmentation_names:
                    mask = res['mask']
                    if not save:
                        del res['mask']
                    if mask is not None and mask.size != 0:
                        croped_batch_seg.append(crop_segmentation(pil_img, box_points, mask))
                        croped_batch_seg_res.append(res)
                        
    add_attributes_in_batch(croped_batch_cans, damage_model, croped_batch_cans_res, 'damage')
    add_attributes_in_batch(croped_batch_cans, fullness_model, croped_batch_cans_res, 'fullness')
    add_attributes_in_batch(croped_batch_seg, garbage_model, croped_batch_seg_res, 'dirtiness')
    
    
def update_result_data(result_data, detection_data, detectCondition):
    for ind, dict_objs in enumerate(result_data):
        if detectCondition:
            dict_objs.update(detection_data[ind])

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
                update_result_data(result_data, detection_data, detectCondition)
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

    return result_data