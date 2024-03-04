import cv2
from fastapi import UploadFile, APIRouter, File, Query
from fastapi.responses import StreamingResponse

from typing import List, Annotated
from detector_segmentator import add_detections_segmentations_on_images
from server_basic_routers import get_crop_detection_segmentation, add_attributes
from server_models import draw_detection_results, add_text_to_image
from server_utils import files_to_batch, wrap_images_as_zip, wrap_video_as_zip

from core import get_crop_detection_segmentation, add_attributes_in_batch, add_attributes, get_all_detection_by_batch, get_difference_garbage_mask, EmptyResualt, get_compose_mask, find_figures, get_difference

import matplotlib.pyplot as plt
import numpy as np

router = APIRouter()

def draw_mask(mask):
    # mask предполагается быть двумерным numpy массивом из булевых значений
    plt.imshow(mask, cmap='gray')  # отрисовка маски в оттенках серого
    plt.axis('off')  # отключение осей координат
    plt.show()
    
def get_compose_mask_and_add_to_list(frame, masks):
    res = get_all_detection_by_batch([frame], False, True, False, True, save_mask=True)[0]
    composed_mask = get_compose_mask(res['garbage'])
    masks.append(composed_mask)
    if len(masks) > 4:
        masks.pop(0)
    

@router.post("/get-t", tags=["test"])
async def get_detect_segmentate_and_images(video_file: UploadFile,
                                           max_frames: int = Query(
                                                100,
                                                description="максимальное количество фреймов в видео"
                                            ),
                                        ):
    try:
        contents = await video_file.read()
    except Exception as e:
        raise HTTPException(status_code=404, detail="Can't read video file. avi type expected. Exception: " + str(e))
        
    with open('temp_video.avi', 'wb') as temp_file:
        temp_file.write(contents)
    
 
    try:
        cap = cv2.VideoCapture('temp_video.avi')
    except Exception as e:
        raise HTTPException(status_code=404, detail="The file was damaged. Exception: " + str(e))
    
    if not cap.isOpened():
        raise HTTPException(status_code=404, detail="Can't open video file may be corrupted")
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    tmp_file_name= 'processed_video.avi'
    
    if fps is None or fps == 0:
        raise HTTPException(status_code=404, detail="Can't get the FPS of the video")
        
    try:
        result_video = cv2.VideoWriter(tmp_file_name, 
                                   cv2.VideoWriter_fourcc(*'MJPG'),
                                   fps, size)
    except:
        raise HTTPException(status_code=404, detail="Can't write video")
    batch_images = []
    frame_count = 0
    masks = []
    masks_shift = []
    
    is_detected = False
    
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        batch_images.append(frame)
        frame_count += 1
        frame_division = int(fps * 3)
        if frame_count % frame_division == 0:
            get_compose_mask_and_add_to_list(frame, masks)
        elif (frame_count + 10) % frame_division == 0:
            get_compose_mask_and_add_to_list(frame, masks_shift)
            if len(masks_shift) == 4:
                mask_dif = get_difference(masks, masks_shift)
                if type(mask_dif) != EmptyResualt:
                    is_detected = is_detected or find_figures(mask_dif)
            
        if len(batch_images) == 32 or (not ret and len(batch_images) > 0):
            for image in batch_images:
                result_video.write(add_text_to_image(image, 'Incorrectly disposed garbage detected') if is_detected else image)
            batch_images = []

    cap.release()
    result_video.release()
    return wrap_video_as_zip(tmp_file_name)
            
    
@router.post("/get-detect-segmentate-and-images", tags=["test"])
async def get_detect_segmentate_and_images(files: List[UploadFile],
                                        detectPeople: bool = Query(
                                            True, 
                                            description="Флаг нужно ли детектить людей"
                                        ),
                                        detectCans: bool = Query(
                                            True, 
                                            description="Флаг нужно ли детектить котнейнеры, места для сбора мусора, объявления"
                                        ),
                                        detectGraffiti: bool = Query(
                                            True, 
                                            description="Флаг нужно ли детектить граффити"
                                        ),
                                        detectGarbage: bool = Query(
                                            True, 
                                            description="Флаг нужно ли детектить мусор"
                                        ),
                                        return_as_zip: bool = Query(
                                            True, 
                                            description="Флаг нужно ли вернуть архив или поток картинок"
                                        ),
                                    ) \
        -> StreamingResponse:
    """
    Принимает батч картинок и выдает батч этих же картинок, но на каждой выделены боксы объектов, маска мусора.
    
    Во время работы применяет алгоритм детекции, сегментации.
    
    Каждый из параметров detectPeople, detectCans, detectGraffiti, 
    и detectGarbage может быть использован для включения или отключения поиска 
    соответствующих объектов в предоставленных изображениях.
    
    Args:
    - files (List[UploadFile]): список загружаемых файлов изображений для анализа.
    - detectCans (bool) = True: если True, будет пытаться обнаружить ['garbage', 'Ad', 'Place', 'Bin', 'Can', 'Container', 'Tank', 'urn'].
    - detectGraffiti (bool) = True: если True, будет пытаться обнаружить ['graffiti'].
    - detectGarbage (bool) = True: если True, будет пытаться обнаружить ['garbage'].
    - detectPeople (bool) = True: если True, будет пытаться обнаружить ['person'].
    - return_as_zip (bool) = True: нужно ли возвращать ответ как архив или как поток выходных картин

    :return: архив или поток картинок
    """
    batch, fnames = await files_to_batch(files, return_fnames=True)
    out_data = get_all_detection_by_batch(batch, detectPeople, detectCans, detectGraffiti, detectGarbage, save_mask=True)
    results = add_detections_segmentations_on_images(batch, out_data, draw_detection_results)

    if return_as_zip:
        return wrap_images_as_zip(results, fnames)
    return StreamingResponse(results, media_type="image/png")

@router.post("/get-detect-segmentate-attributes-and-images", tags=["test"])
async def get_detect_segmentate_attributes_and_images(files: List[UploadFile],
                                        detectPeople: bool = Query(
                                            True, 
                                            description="Флаг нужно ли детектить людей"
                                        ),
                                        detectCans: bool = Query(
                                            True, 
                                            description="Флаг нужно ли детектить котнейнеры, места для сбора мусора, объявления"
                                        ),
                                        detectGraffiti: bool = Query(
                                            True, 
                                            description="Флаг нужно ли детектить граффити"
                                        ),
                                        detectGarbage: bool = Query(
                                            True, 
                                            description="Флаг нужно ли детектить мусор"
                                        ),
                                        return_as_zip: bool = Query(
                                            True, 
                                            description="Флаг нужно ли вернуть архив или поток картинок"
                                        ),
                                    ) \
        -> StreamingResponse:
    """
    Принимает батч картинок и выдает батч этих же картинок, но на каждой выделены боксы объектов, маска мусора и добавлены иконки атрибутов.
    
    Во время работы применяет алгоритм детекции, сегментации и классификации.

    Каждый из параметров detectPeople, detectCans, detectGraffiti, 
    и detectGarbage может быть использован для включения или отключения поиска 
    соответствующих объектов в предоставленных изображениях.
    
    Контейнеры для мусора имеют следующие атрибуты:
    Классфикация по принципу повреждений ['broken', 'flipped', 'ok']
        broken - найдены сильные повреждения на контейнере.
        flipped - перевернут.
    Классфикация по принципу наполенености мусором ['full', 'half', 'empty']
        
    Мусор имеют следующие атрибуты:
    Классфикация по степени замусоренности ['TKO', 'KGO', 'bulk']
        TKO - мелкий мусор.
        KGO - крупногабаритные отходы -  твердые коммунальные отходы, размер которых не позволяет осуществить их складирование в контейнерах.
        Bulk - навал мусора -  скопление твердых бытовых отходов и крупногабаритного мусора, по объему, не превышающему 1 м3.
    
    Args:
    - files (List[UploadFile]): список загружаемых файлов изображений для анализа.
    - detectCans (bool) = True: если True, будет пытаться обнаружить ['garbage', 'Ad', 'Place', 'Bin', 'Can', 'Container', 'Tank', 'urn'].
    - detectGraffiti (bool) = True: если True, будет пытаться обнаружить ['graffiti'].
    - detectGarbage (bool) = True: если True, будет пытаться обнаружить ['garbage'].
    - detectPeople (bool) = True: если True, будет пытаться обнаружить ['person'].
    - return_as_zip (bool) = True: нужно ли возвращать ответ как архив или как поток выходных картин

    :return: архив или поток картинок
    """
    batch, fnames = await files_to_batch(files, return_fnames=True)
    detection_segmentation_result = get_all_detection_by_batch(batch, detectPeople, detectCans, detectGraffiti, detectGarbage, save_mask=True)
    add_attributes(batch, detection_segmentation_result, True)
    results = add_detections_segmentations_on_images(batch, detection_segmentation_result, draw_detection_results)
    
    if return_as_zip:
        return wrap_images_as_zip(results, fnames)
    return StreamingResponse(results, media_type="image/png")
    
@router.post("/detect-segmentate-crop", tags=["test"])
async def detect_segmentate_crop(files: List[UploadFile],
                                        detectPeople: bool = Query(
                                            True, 
                                            description="Флаг нужно ли детектить людей"
                                        ),
                                        detectCans: bool = Query(
                                            True, 
                                            description="Флаг нужно ли детектить котнейнеры, места для сбора мусора, объявления"
                                        ),
                                        detectGraffiti: bool = Query(
                                            True, 
                                            description="Флаг нужно ли детектить граффити"
                                        ),
                                        detectGarbage: bool = Query(
                                            True, 
                                            description="Флаг нужно ли детектить мусор"
                                        ),
                                        return_as_zip: bool = Query(
                                            True, 
                                            description="Флаг нужно ли вернуть архив или поток картинок"
                                        ),
                                    ):
    """
    Принимает батч картинок и выдает список вырезанных детекций и сегментаций.
    
    Каждый из параметров detectPeople, detectCans, detectGraffiti, 
    и detectGarbage может быть использован для включения или отключения поиска 
    соответствующих объектов в предоставленных изображениях.
    
    Args:
    - files (List[UploadFile]): список загружаемых файлов изображений для анализа.
    - detectCans (bool) = True: если True, будет пытаться обнаружить ['garbage', 'Ad', 'Place', 'Bin', 'Can', 'Container', 'Tank', 'urn'].
    - detectGraffiti (bool) = True: если True, будет пытаться обнаружить ['graffiti'].
    - detectGarbage (bool) = True: если True, будет пытаться обнаружить ['garbage'].
    - detectPeople (bool) = True: если True, будет пытаться обнаружить ['person'].
    - return_as_zip (bool) = True: нужно ли возвращать ответ как архив или как поток выходных картинок.

    Во время работы применяет алгоритм детекции и сегментации.

    Args:
        files: батч картинок

        return_as_zip: нужно ли возвращать ответ как архив или как поток выходных картинок.
        По умолчанию return_as_zip=True.

    Returns:
        res: архив или поток картинок
    """
    batch = await files_to_batch(files, return_fnames=False)
    result = get_all_detection_by_batch(batch, detectPeople, detectCans, detectGraffiti, detectGarbage, save_mask=True)
    results = get_crop_detection_segmentation(batch, result, True, True)
    if return_as_zip:
        return wrap_images_as_zip(results)
    return StreamingResponse(results, media_type="image/png")
    
def process_batch_images(batch, detectPeople, detectCans, detectGraffiti, detectGarbage):
    detection_segmentation_result = get_all_detection_by_batch(batch, detectPeople, detectCans, detectGraffiti, detectGarbage, save_mask=True)
    add_attributes(batch, detection_segmentation_result, True)
    return add_detections_segmentations_on_images(batch, detection_segmentation_result, draw_detection_results, False)
    

    
@router.post("/process-video-with-detection-only", tags=["test"])
async def process_video_with_detection_only(video_file: UploadFile,
                                            detectPeople: bool = Query(
                                                True, 
                                                description="Флаг нужно ли детектить людей"
                                            ),
                                            detectCans: bool = Query(
                                                True, 
                                                description="Флаг нужно ли детектить котнейнеры, места для сбора мусора, объявления"
                                            ),
                                            detectGraffiti: bool = Query(
                                                True, 
                                                description="Флаг нужно ли детектить граффити"
                                            ),
                                            detectGarbage: bool = Query(
                                                True, 
                                                description="Флаг нужно ли детектить мусор"
                                            ),
                                            max_frames: int = Query(
                                                100,
                                                description="максимальное количество фреймов в видео"
                                            ),
                                        ):
    """
    Принимает видео и максимальное количество фреймов и выдает видео на котором выделены объекты, маски и указана классификация объектов.
    
    Каждый из параметров detectPeople, detectCans, detectGraffiti, 
    и detectGarbage может быть использован для включения или отключения поиска 
    соответствующих объектов в предоставленных изображениях.
    
    Контейнеры для мусора имеют следующие атрибуты:
    Классфикация по принципу повреждений ['broken', 'flipped', 'ok']
        broken - найдены сильные повреждения на контейнере.
        flipped - перевернут.
    Классфикация по принципу наполенености мусором ['full', 'half', 'empty']
        
    Мусор имеют следующие атрибуты:
    Классфикация по степени замусоренности ['TKO', 'KGO', 'bulk']
        TKO - мелкий мусор.
        KGO - крупногабаритные отходы -  твердые коммунальные отходы, размер которых не позволяет осуществить их складирование в контейнерах.
        Bulk - навал мусора -  скопление твердых бытовых отходов и крупногабаритного мусора, по объему, не превышающему 1 м3.
    
    Args:
    - video_file (UploadFile): видео в формате avi.
    - detectCans (bool) = True: если True, будет пытаться обнаружить ['garbage', 'Ad', 'Place', 'Bin', 'Can', 'Container', 'Tank', 'urn'].
    - detectGraffiti (bool) = True: если True, будет пытаться обнаружить ['graffiti'].
    - detectGarbage (bool) = True: если True, будет пытаться обнаружить ['garbage'].
    - detectPeople (bool) = True: если True, будет пытаться обнаружить ['person'].
    - max_frames (int) = 100: максимальное количество фреймов в видео
    Returns:

        Zip файл в котором лежит обработаное видео
    """
    try:
        contents = await video_file.read()
    except:
        raise HTTPException(status_code=404, detail="Can't read video file. avi type expected")
    with open('temp_video.avi', 'wb') as temp_file:
        temp_file.write(contents)
    
 
    try:
        cap = cv2.VideoCapture('temp_video.avi')
    except:
        raise HTTPException(status_code=404, detail="The file was damaged")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (frame_width, frame_height)
    tmp_file_name= 'processed_video.avi'
    try:
        result_video = cv2.VideoWriter(tmp_file_name, 
                                   cv2.VideoWriter_fourcc(*'MJPG'),
                                   10, size)
    except:
        raise HTTPException(status_code=404, detail="Can't write video")

    batch_images = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        batch_images.append(frame)
        frame_count += 1

        if len(batch_images) == 32 or (not ret and len(batch_images) > 0):
            processed_batch = process_batch_images(batch_images, detectPeople, detectCans, detectGraffiti, detectGarbage)
            for processed_frame in processed_batch:
                result_video.write(processed_frame)
            batch_images = []

    cap.release()
    result_video.release()
    return wrap_video_as_zip(tmp_file_name)

