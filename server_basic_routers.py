from io import BytesIO
from typing import List, Annotated

import numpy as np
from PIL import Image
from fastapi import UploadFile, APIRouter, File
from tqdm import tqdm

from align_utils import crop_img, crop_cans, crop_segmentation
from detector_segmentator import PersonDetector, GraffitiDetector, GarbageSegmentator, CansDetector
from server_models import cans, people, garbage, graffiti, garbage_model, fullness_model, damage_model, cans_names, segmentation_names
from server_utils import files_to_batch, normilise_batch_cv2

router = APIRouter()

@router.post("/get-garbage-attributes", tags=["basic"])
async def get_garbage_attributes(files: Annotated[List[UploadFile], File(description="Батч карт инок")]):
    """
        Принимает батч картинок и выдает батч классификации.
        Классфикация по степени замусоренности ['TKO', 'KGO', 'bulk']
        TKO - мелкий мусор.
        KGO - крупногабаритные отходы -  твердые коммунальные отходы, размер которых не позволяет осуществить их складирование в контейнерах.
        Bulk - навал мусора -  скопление твердых бытовых отходов и крупногабаритного мусора, по объему, не превышающему 1 м3.
        
        Returns:

            dict: [[garbage_attr]]
    """
    batch = await files_to_batch(files, for_attr=True)
    out_data = garbage_model.run(batch)
    return {"result": out_data}
    
@router.post("/get-damage-attributes", tags=["basic"])
async def get_damage_attributes(files: Annotated[List[UploadFile], File(description="Батч картинок")]):
    """
        Принимает батч картинок и выдает батч классификации.
        Классфикация по принципу повреждений ['broken', 'flipped', 'ok']
        broken - найдены сильные повреждения на контейнере.
        flipped - перевернут.
        
        Returns:

            dict: [[damage_attr]]
    """
    batch = await files_to_batch(files, for_attr=True)
    out_data = damage_model.run(batch)
    return {"result": out_data}
    
@router.post("/get-fullnes-attributes", tags=["basic"])
async def get_fullnes_attributes(files: Annotated[List[UploadFile], File(description="Батч картинок")]):
    """
        Принимает батч картинок и выдает батч классификации.
        Классфикация по принципу наполенености мусором ['full', 'half', 'empty']
        
        Returns:

            dict: [[fullness_attr]]
    """
    batch = await files_to_batch(files, for_attr=True)
    out_data = fullness_model.run(batch)
    return {"result": out_data}
    
@router.post("/detect-cans", tags=["basic"])
async def get_detected_cans(files: Annotated[List[UploadFile], File(description="Батч картинок")]):
    """
    Принимает батч картинок и выдает батч детекций ['garbage', 'Ad', 'Place', 'Bin', 'Can', 'Container', 'Tank', 'urn'].
    Каждая детекция - это [уверенность, список BBox'ов]

    Returns:

        dict: [{'object_name':[{box_points}]}]
    """
    batch = await files_to_batch(files)
    out_data = cans.run(batch)
    return {"result": out_data}   
    
@router.post("/detect-graffiti", tags=["basic"])
async def get_detected_graffiti(files: Annotated[List[UploadFile], File(description="Батч картинок")]):
    """
    Принимает батч картинок и выдает батч детекций ['graffiti'].
    Каждая детекция - это [уверенность, список BBox'ов]

    Returns:

        dict: [{'object_name':[{box_points}]}]
    """
    batch = await files_to_batch(files)
    out_data = graffiti.run(batch)
    return {"result": out_data}  
    
@router.post("/detect-people", tags=["basic"])
async def get_detected_people(files: Annotated[List[UploadFile], File(description="Батч картинок")]):
    """
    Принимает батч картинок и выдает батч детекций ['people'].
    Каждая детекция - это [уверенность, список BBox'ов]

    Returns:

        dict: [{'object_name':[{box_points}]}]
    """
    batch = await files_to_batch(files)
    out_data = people.run(batch)
    return {"result": out_data}  
    
@router.post("/segment-garbage", tags=["basic"])
async def get_segmented_garbage(files: Annotated[List[UploadFile], File(description="Батч картинок")]):
    """
    Принимает батч картинок и выдает батч сегментации ['garbage'].
    Каждая детекция - это [уверенность, список BBox'ов, саисок масок]

    Returns:

        dict: [{'object_name':[{box_points}]}]
    """
    batch = await files_to_batch(files)
    out_data = garbage.run(batch)
    return {"result": out_data}  

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
