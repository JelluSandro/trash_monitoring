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
        
async def video(video_file, max_frames, need_find_disposal=False, detectPeople=False, detectCans=False, detectGraffiti=False, detectGarbage=False, garbageAttribute=False, fullnesAttribute=False, damageAttribute=False):
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
        
        if need_find_disposal:
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
            if detectPeople or detectCans or detectGraffiti or detectGarbage:
                processed_batch = process_batch_images(batch_images, detectPeople, detectCans, detectGraffiti, detectGarbage, garbageAttribute, fullnesAttribute, damageAttribute)
            else:
                processed_batch = batch_images
            for image in processed_batch:
                result_video.write(add_text_to_image(image, 'Incorrectly disposed garbage detected') if is_detected else image)
            batch_images = []

    cap.release()
    result_video.release()
    return wrap_video_as_zip(tmp_file_name)
    
@router.post("/detect-and-draw-detection-on-images", tags=["test"])
async def detect_and_draw_detection_on_images(files: List[UploadFile],
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
    Принимает на вход фотографии в формате, поддерживающемся библиотекой cv2, архивы формата zip, внутри которых содержатся фотографии подобного типа.
    
    Функция обнаруживает на переданных изображениях объекты.
    Для каждого обнаруженного объекта на переданном изображении рисуется ограничивающая рамка, цвет рамки зависит от типа объекта.
    
    * Реклама - Ad - цвет Синий.
    * Люди - person - цвет Черный.
    * Граффити - graffiti - цвет Лаймовый.
    * Мусор - garbage - цвет чайный.
    * Площадка - Place - цвет Фуксия.
    * Контейнеры для спец. мусора - Bin - цвет Красный.
    * Железные контейнеры - Can - цвет Темно-бордовый.
    * Пластиковые контейнеры - Container - цвет Желтый.
    * Танкеры - Tank - цвет Оливковый.
    * Урны - urn - цвет Зеленый.
    
    Каждый из параметров detectPeople, detectCans, detectGraffiti и detectGarbage может быть использован для включения или отключения поиска соответствующих объектов в предоставленных изображениях.
    
    ### Аргументы:
    
    * **files** (List[UploadFile]): список загружаемых файлов изображений для анализа.
    * **detectCans** (bool) = True: если True, будет пытаться обнаружить следующие объекты: контейнеры, площадки, объявления.
    * **detectGraffiti** (bool) = True: если True, будет пытаться обнаружить граффити.
    * **detectGarbage** (bool) = True: если True, будет пытаться обнаружить мусор.
    * **detectPeople** (bool) = True: если True, будет пытаться обнаружить людей.
    * **return_as_zip** (bool) = True: нужно ли возвращать ответ как архив или как поток выходных картин.
    
    ### Возвращаемое значение:
    
    Архив или поток картинок, на которых выделены ограничивающими рамками найденные объекты.
    """
    batch, fnames = await files_to_batch(files, return_fnames=True)
    out_data = get_all_detection_by_batch(batch, detectPeople, detectCans, detectGraffiti, detectGarbage, save_mask=True)
    results = add_detections_segmentations_on_images(batch, out_data, draw_detection_results)

    if return_as_zip:
        return wrap_images_as_zip(results, fnames)
    return StreamingResponse(results, media_type="image/png")

@router.post("/detect-classify-сontainerStatus-containerFilling-garbageType-and-draw-on-image", tags=["test"])
async def detect_classify_сontainerStatus_containerFilling_garbageType_and_draw_on_image(files: List[UploadFile],
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
                                        garbageType: bool = Query(
                                            True, 
                                            description="Флаг нужно ли определять мусор"
                                        ),
                                        containerFilling: bool = Query(
                                            True, 
                                            description="Флаг нужно ли определять насколько заполнены контейнеры"
                                        ),
                                        containerStatus: bool = Query(
                                            True, 
                                            description="Флаг нужно ли определять повреждены ли контейнеры"
                                        ),
                                        return_as_zip: bool = Query(
                                            True, 
                                            description="Флаг нужно ли вернуть архив или поток картинок"
                                        ),
                                    ) \
        -> StreamingResponse:
    """
    Принимает на вход фотографии в формате, поддерживающемся библиотекой cv2, архивы формата zip внутри которых содержатся фотографии подобного типа.
    
    Функция обнаруживает на переданных изображениях объекты.
    Для каждого обнаруженного объекта на переданном изображении рисуется ограничивающая рамка, цвет рамки зависит от типа объекта.
    В зависимости от класса объекта в правом верхнем углу ограничивающей рамки рисуется иконка класса.
    
    * Реклама - Ad - цвет Синий.
    * Люди - person - цвет Черный.
    * Граффити - graffiti - цвет Лаймовый.
    * Мусор - garbage - цвет чайный.
    * Площадка - Place - цвет Фуксия.
    * Контейнеры для спец. мусора - Bin - цвет Красный.
    * Железные контейнеры - Can - цвет Темно-бордовый.
    * Пластиковые контейнеры - Container - цвет Желтый.
    * Танкеры - Tank - цвет Оливковый.
    * Урны - urn - цвет Зеленый.

    Функция распознаёт контейнеры и относит их к одной из следующих категорий:

    * нормальное состояние — ok - контейнер цел и не имеет видимых повреждений.
    * сильные повреждения — broken - на контейнере есть значительные дефекты, которые могут повлиять на его целостность.
    * перевёрнут — flipped - контейнер находится в перевёрнутом положении.

    Также функция определяет степень заполненности контейнера:

    * пустой — empty - контейнер не содержит мусора.
    * есть место — half - контейнер заполнен, но есть место для нового мусора.
    * заполнен — full - в контейнере нет места для нового мусора.
  
    Функция определяет вид мусора:
    
    * мелкий мусор - TKO.
    * крупногабаритные отходы — KGO — твёрдые коммунальные отходы, размер которых не позволяет осуществить их складирование в контейнерах.
    * навал мусора - Bulk - скопление твердых бытовых отходов и крупногабаритного мусора, по объему, не превышающему 1 м3. 
    
    Каждый из параметров detectPeople, detectCans, detectGraffiti и detectGarbage может быть использован для включения или отключения поиска соответствующих объектов в предоставленных изображениях.
    
    Каждый из параметров garbageType, containerFilling, containerStatus может быть использован для включения или отключения поиска соответствующих аттрибутов в предоставленных изображениях.
    
    ### Аргументы:
    
    * **files** (List[UploadFile]): список загружаемых файлов изображений для анализа.
    * **detectCans** (bool) = True: если True, будет пытаться обнаружить следующие объекты: контейнеры, площадки, объявления.
    * **detectGraffiti** (bool) = True: если True, будет пытаться обнаружить граффити.
    * **detectGarbage** (bool) = True: если True, будет пытаться обнаружить мусор.
    * **detectPeople** (bool) = True: если True, будет пытаться обнаружить людей.
    * **garbageType** (bool) = True: если True, будет определять тип мусора.
    * **containerFilling** (bool) = True: если True, будет определять заполненность контейнеров.
    * **containerStatus** (bool) = True: если True, будет определять состояние контейнеров.
    
    ### Возвращаемое значение:
    
    Архив или поток картинок.
    """
    batch, fnames = await files_to_batch(files, return_fnames=True)
    detection_segmentation_result = get_all_detection_by_batch(batch, detectPeople, detectCans, detectGraffiti, detectGarbage, save_mask=True)
    add_attributes(batch, detection_segmentation_result, garbageType, containerFilling, containerStatus, True)
    results = add_detections_segmentations_on_images(batch, detection_segmentation_result, draw_detection_results)
    
    if return_as_zip:
        return wrap_images_as_zip(results, fnames)
    return StreamingResponse(results, media_type="image/png")
    
@router.post("/detect-segmentate-crop", tags=["test"])
async def get_extracted_objects_from_images(files: List[UploadFile],
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
    Принимает на вход фотографии в формате, поддерживающемся библиотекой cv2, архивы формата zip внутри которых содержатся фотографии подобного типа.
    
    Функция вырезает объекты и маски из изображения и возвращает их в виде архива или потока картинок.
    
    Вырезаемые объекты: 
    
    * Граффити - graffiti.
    * Реклама - Ad.
    * Люди - person.
    * Мусор - garbage.
    * Контейнерная площадка - Place.
    * Контейнеры для спец. мусора - Bin.
    * Железные контейнеры - Can.
    * Пластиковые контейнеры - Container.
    * Танкеры - Tank.
    * Урны - urn.
    
    Каждый из параметров detectPeople, detectCans, detectGraffiti и detectGarbage может быть использован для включения или отключения поиска соответствующих объектов в предоставленных изображениях.
    

    ### Аргументы:
    
    * **files** (List[UploadFile]): список загружаемых файлов изображений для анализа.
    * **detectCans** (bool) = True: если True, будет пытаться обнаружить следующие объекты: контейнеры, площадки, объявления.
    * **detectGraffiti** (bool) = True: если True, будет пытаться обнаружить граффити.
    * **detectGarbage** (bool) = True: если True, будет пытаться обнаружить мусор.
    * **detectPeople** (bool) = True: если True, будет пытаться обнаружить людей.
        
    ### Возвращаемое значение:
    
    Архив или поток картинок.
    """
    batch = await files_to_batch(files, return_fnames=False)
    result = get_all_detection_by_batch(batch, detectPeople, detectCans, detectGraffiti, detectGarbage, save_mask=True)
    results = get_crop_detection_segmentation(batch, result, True, True)
    if return_as_zip:
        return wrap_images_as_zip(results)
    return StreamingResponse(results, media_type="image/png")
    
def process_batch_images(batch, detectPeople, detectCans, detectGraffiti, detectGarbage, garbageAttribute, fullnesAttribute, damageAttribute):
    detection_segmentation_result = get_all_detection_by_batch(batch, detectPeople, detectCans, detectGraffiti, detectGarbage, save_mask=True)
    add_attributes(batch, detection_segmentation_result, garbageAttribute, fullnesAttribute, damageAttribute, True)
    return add_detections_segmentations_on_images(batch, detection_segmentation_result, draw_detection_results, False)
    

    
@router.post("/process-video-detect-classify-and-draw", tags=["test"])
async def process_video_detect_classify_and_draw(video_file: UploadFile,
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
                                            garbageType: bool = Query(
                                                True, 
                                                description="Флаг нужно ли определять мусор"
                                            ),
                                            containerFilly: bool = Query(
                                                True, 
                                                description="Флаг нужно ли определять насколько заполнены контейнеры"
                                            ),
                                            containerStatus: bool = Query(
                                                True, 
                                                description="Флаг нужно ли определять повреждены ли контейнеры"
                                            ),
                                            max_frames: int = Query(
                                                100,
                                                description="максимальное количество фреймов в видео"
                                            ),
                                        ):
    """
    Принимает видео и максимальное количество фреймов и выдает видео на котором выделены объекты, маски и классы объектов.
    
    Функция обнаруживает на изображениях такие объекты как: 
    
    * Граффити - graffiti.
    * Реклама - Ad.
    * Люди - person.
    * Мусор - garbage.
    * Контейнерная площадка - Place.
    * Контейнеры для спец. мусора - Bin.
    * Железные контейнеры - Can.
    * Пластиковые контейнеры - Container.
    * Танкеры - Tank.
    * Урны - urn.
    
    Функция распознаёт контейнеры и относит их к одной из следующих категорий:

    * нормальное состояние — ok - контейнер цел и не имеет видимых повреждений.
    * сильные повреждения — broken - на контейнере есть значительные дефекты, которые могут повлиять на его целостность.
    * перевёрнут — flipped - контейнер находится в перевёрнутом положении.

    Также функция определяет степень заполненности контейнера:

    * пустой — empty - контейнер не содержит мусора.
    * есть место — half - контейнер заполнен, но есть место для нового мусора.
    * заполнен — full - в контейнере нет места для нового мусора.
  
    Функция определяет вид мусора:
    
    * мелкий мусор - TKO.
    * крупногабаритные отходы — KGO — твёрдые коммунальные отходы, размер которых не позволяет осуществить их складирование в контейнерах.
    * навал мусора - Bulk - скопление твердых бытовых отходов и крупногабаритного мусора, по объему, не превышающему 1 м3. 

    Каждый из параметров detectPeople, detectCans, detectGraffiti и detectGarbage может быть использован для включения или отключения поиска соответствующих объектов в предоставленных изображениях.
    
    Каждый из параметров garbageType, containerFilling, сontainerStatus может быть использован для включения или отключения поиска соответствующих аттрибутов в предоставленных изображениях.

    ### Аргументы:
    
    * **video_file** (UploadFile): видео в формате avi.
    * **detectCans** (bool) = True: если True, будет пытаться обнаружить следующие объекты: контейнеры, площадки, объявления.
    * **detectGraffiti** (bool) = True: если True, будет пытаться обнаружить граффити.
    * **detectGarbage** (bool) = True: если True, будет пытаться обнаружить  мусор.
    * **detectPeople** (bool) = True: если True, будет пытаться обнаружить людей.
    * **garbageType** (bool) = True: если True, будет определять тип мусора.
    * **containerFilly** (bool) = True: если True, будет определять заполненность контейнеров.
    * **containerStatus** (bool) = True: если True, будет определять состояние контейнеров.
    * **max_frames** (int) = 100: максимальное количество фреймов в видео
    
    ### Возвращаемое значение:

    Zip файл в котором лежит обработанное видео.
    """
    return await video(video_file, max_frames, False, detectPeople, detectCans, detectGraffiti, detectGarbage, garbageType, containerFilly, containerStatus)


@router.post("/detect-incorrectly-disposed-garbage", tags=["test"])
async def detect_incorrectly_disposed_garbage(video_file: UploadFile,
                                           max_frames: int = Query(
                                                100,
                                                description="максимальное количество фреймов в видео"
                                            ),
                                        ):
    """
    Функция обнаруживает и маркирует неправильно выброшенный мусор.

    Функция принимает видео или видеопоток в качестве входных данных и возвращает обработанные данные.

    В процессе обработки данных программа анализирует видео на предмет неправильно выброшенного мусора. 
    Если программа обнаруживает такой случай, то в правом верхнем углу видео добавляется надпись 
    *incorrectly disposed garbage detected*. Надпись появляется в тот момент, когда программа считает, что произошло неправильное выбрасывание мусора.

    Если неправильно выброшенный мусор не обнаружен, то функция возвращает исходное видео без изменений.
        
    ### Аргументы:
    
    * **video_file** (UploadFile): видео в формате avi.
    * **max_frames** (int) = 100: максимальное количество фреймов в видео
        
    ### Возвращаемое значение:
    
    Zip файл в котором лежит обработанное видео.
    """
    return await video(video_file, max_frames, need_find_disposal=True, detectPeople=False, detectCans=False, detectGraffiti=False, detectGarbage=False)
