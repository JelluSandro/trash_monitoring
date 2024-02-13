import math
import tempfile
from typing import List

import cv2
import numpy as np
from fastapi import UploadFile, APIRouter
from fastapi.responses import StreamingResponse
from tqdm import tqdm

from fastapi import UploadFile, APIRouter, File

from typing import List, Annotated
from align_utils import draw_rectangles_on_cans
from detector_segmentator import add_detections_segmentations_on_images
from server_basic_routers import get_crop_detection_segmentation, add_attributes
from server_models import draw_detection_results
from server_models import cans, people, garbage, graffiti, garbage_model, fullness_model, damage_model, segmentation_names, cans_names
from server_utils import files_to_batch, wrap_images_as_zip, wrap_video_as_zip

router = APIRouter()

    
@router.post("/get-detect-segmentate-and-images", tags=["test"])
async def get_detect_segmentate_and_images(files: List[UploadFile],
                                        return_as_zip: bool = True) \
        -> StreamingResponse:
    """
    Принимает батч картинок и выдает батч этих же картинок, но на каждой выделены боксы объектов, маска мусора.
    
    Во время работы применяет алгоритм детекции, сегментации.

    :param files: батч картинок

    :param return_as_zip: нужно ли возвращать ответ как архив или как поток выходных картинок.
    По умолчанию return_as_zip=True.

    :return: архив или поток картинок
    """
    batch, fnames = await files_to_batch(files, return_fnames=True)
    out_data = get_all_detection_by_batch(batch, save_mask=True)
    results = add_detections_segmentations_on_images(batch, out_data, draw_detection_results)

    if return_as_zip:
        return wrap_images_as_zip(results, fnames)
    return StreamingResponse(results, media_type="image/png")

@router.post("/get-detect-segmentate-attributes-and-images", tags=["test"])
async def get_detect_segmentate_attributes_and_images(files: List[UploadFile],
                                        return_as_zip: bool = True) \
        -> StreamingResponse:
    """
    Принимает батч картинок и выдает батч этих же картинок, но на каждой выделены боксы объектов, маска мусора и добавлены иконки атрибутов.
    
    Во время работы применяет алгоритм детекции, сегментации и классификации.

    :param files: батч картинок

    :param return_as_zip: нужно ли возвращать ответ как архив или как поток выходных картинок.
    По умолчанию return_as_zip=True.

    :return: архив или поток картинок
    """
    batch, fnames = await files_to_batch(files, return_fnames=True)
    detection_segmentation_result = get_all_detection_by_batch(batch, save_mask=True)
    add_attributes(batch, detection_segmentation_result, True)
    results = add_detections_segmentations_on_images(batch, detection_segmentation_result, draw_detection_results)
    
    if return_as_zip:
        return wrap_images_as_zip(results, fnames)
    return StreamingResponse(results, media_type="image/png")

def get_all_detection_by_batch(batch, save_mask=False):
    graffiti_data = graffiti.run(batch)
    cans_data = cans.run(batch) 
    people_data = people.run(batch)
    edited_batch = []
    for (image, info) in tqdm(zip(batch, cans_data)):
        edited_image = image.copy()
        for key in cans_names:
            for res in info[key]:
                box_points = res['box_points']
                draw_rectangles_on_cans(edited_image, box_points)
        edited_batch.append(edited_image)
    garbage_data = garbage.run(edited_batch, save_mask)
    for ind, dict_objs in enumerate(graffiti_data):
        dict_objs.update(cans_data[ind])
        dict_objs.update(people_data[ind])
        dict_objs.update(garbage_data[ind])
    return graffiti_data
    
    
@router.post("/detect-segmentate-crop", tags=["test"])
async def detect_segmentate_crop(files: List[UploadFile],
                                        return_as_zip: bool = True,
                                        crop_only_attr: bool = True):
    """
    Принимает батч картинок и выдает список вырезанных детекций и сегментаций.

    Во время работы применяет алгоритм детекции и сегментации.

    Args:
        files: батч картинок

        return_as_zip: нужно ли возвращать ответ как архив или как поток выходных картинок.
        По умолчанию return_as_zip=True.

    Returns:
        res: архив или поток картинок
    """
    batch = await files_to_batch(files, return_fnames=False)
    result = get_all_detection_by_batch(batch, save_mask=True)
    results = get_crop_detection_segmentation(batch, result, True, True)
    if return_as_zip:
        return wrap_images_as_zip(results)
    return StreamingResponse(results, media_type="image/png")

@router.post("/get_all_detection", tags=["test"])
async def get_all_detection(files: Annotated[List[UploadFile], File(description="Батч картинок")]):
    """
    Принимает батч картинок и выдает батч детекций.
    Каждая детекция - это словарь {'название объекта':[список BBox'ов]}

    Returns:

        dict: [{'object_name':[{box_points}]}]
    """
    batch = await files_to_batch(files)
    return  {"result": get_all_detection_by_batch(batch) }
    
@router.post("/get_all_detection_attributes", tags=["test"])
async def get_all_detection_attributes(files: Annotated[List[UploadFile], File(description="Батч картинок")]):
    """
    Принимает батч картинок и выдает батч детекций и атрибутов.
    Каждая детекция - это словарь {'название объекта':[список BBox'ов, Название класса: атрибут]}

    Returns:

        dict: [{'object_name':[{box_points, obj_class}]}]
    """
    batch = await files_to_batch(files, return_fnames=False)
    result = get_all_detection_by_batch(batch, save_mask=True)
    add_attributes(batch, result)
    return  {"result": result } 
    
def process_batch_images(batch):
    detection_segmentation_result = get_all_detection_by_batch(batch, save_mask=True)
    add_attributes(batch, detection_segmentation_result, True)
    return add_detections_segmentations_on_images(batch, detection_segmentation_result, draw_detection_results, False)
    
@router.post("/process-video-with-detection-only", tags=["test"])
async def process_video_with_detection_only(video_file: UploadFile,
                                            max_frames: int = 100):
    """
    Принимает видео и максимальное количество фреймов и выдает видео на котором выделены объекты, маски и указана классификация объектов.

    Returns:

        Zip файл в котором лежит обработаное видео
    """
    contents = await video_file.read()
    with open('temp_video.avi', 'wb') as temp_file:
        temp_file.write(contents)

    cap = cv2.VideoCapture('temp_video.avi')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    tmp_file_name= 'processed_video.avi'
    result_video = cv2.VideoWriter(tmp_file_name, 
                                   cv2.VideoWriter_fourcc(*'MJPG'),
                                   10, size)

    batch_images = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        batch_images.append(frame)
        frame_count += 1

        if len(batch_images) == 32 or (not ret and len(batch_images) > 0):
            processed_batch = process_batch_images(batch_images)
            for processed_frame in processed_batch:
                result_video.write(processed_frame)
            batch_images = []

    cap.release()
    result_video.release()
    return wrap_video_as_zip(tmp_file_name)
