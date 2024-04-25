from typing import List, Annotated

from fastapi import UploadFile, APIRouter, File, Query
from server_utils import files_to_batch, normilise_batch_cv2
from core import get_crop_detection_segmentation, add_attributes, get_all_detection_by_batch

router = APIRouter()

@router.post("/get_all_detection", tags=["basic"])
async def get_all_detection(files: Annotated[List[UploadFile], File(description="Батч картинок")],
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
                        ):
    """
    Эндпоинт анализирует пакет изображений и возвращает детекции для объектов, 
    указанных в параметрах запроса. Каждый из параметров detectPeople, detectCans, detectGraffiti, 
    и detectGarbage может быть использован для включения или отключения поиска 
    соответствующих объектов в предоставленных изображениях.
    
    Args:
    - files (List[UploadFile]): список загружаемых файлов изображений для анализа.
    - detectCans (bool) = True: если True, будет пытаться обнаружить ['garbage', 'Ad', 'Place', 'Bin', 'Can', 'Container', 'Tank', 'urn'].
    - detectGraffiti (bool) = True: если True, будет пытаться обнаружить ['graffiti'].
    - detectGarbage (bool) = True: если True, будет пытаться обнаружить ['garbage'].
    - detectPeople (bool) = True: если True, будет пытаться обнаружить ['person'].

    Каждая детекция - это словарь {'название объекта':[список BBox'ов]}

    Returns:

        {result: {'object_name':[{box_points}]}}
    """
    batch = await files_to_batch(files)
    return  {"result": get_all_detection_by_batch(batch, detectPeople, detectCans, detectGraffiti, detectGarbage,) }
    
@router.post("/get_all_detection_attributes", tags=["basic"])
async def get_all_detection_attributes(files: Annotated[List[UploadFile], File(description="Батч картинок")],
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
                            garbageAttribute: bool = Query(
                                True, 
                                description="Флаг нужно ли определять мусор"
                            ),
                            fullnesAttribute: bool = Query(
                                True, 
                                description="Флаг нужно ли определять насколько заполнены контейнеры"
                            ),
                            damageAttribute: bool = Query(
                                True, 
                                description="Флаг нужно ли определять повреждены ли контейнеры"
                            ),
                        ):
    """
    Эндпоинт анализирует пакет изображений и возвращает детекции и набор атрибутов для объектов, 
    указанных в параметрах запроса. Каждый из параметров detectPeople, detectCans, detectGraffiti, 
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
    - garbageAttribute (bool) = True: если True, будет определять тип мусора.
    - fullnesAttribute (bool) = True: если True, будет определять заполненость контейнеров.
    - damageAttribute (bool) = True: если True, будет определять состояние контейнеров.
    
    Каждая детекция - это словарь {'название объекта':[список BBox'ов, Название класса: атрибут]}

    Returns:

        dict: [{'object_name':[{box_points, obj_class}]}]
    """
    batch = await files_to_batch(files, return_fnames=False)
    result = get_all_detection_by_batch(batch, detectPeople, detectCans, detectGraffiti, detectGarbage, save_mask=True)
    add_attributes(batch, result, garbageAttribute, fullnesAttribute, damageAttribute)
    return  {"result": result } 