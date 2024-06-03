from typing import List, Annotated

from fastapi import UploadFile, APIRouter, File, Query
from server_utils import files_to_batch, normilise_batch_cv2
from core import get_crop_detection_segmentation, add_attributes, get_all_detection_by_batch

router = APIRouter()

@router.post("/detect-people-cans-graffiti-garbage", tags=["basic"])
async def detect_people_cans_graffiti_garbage(files: Annotated[List[UploadFile], File(description="Батч картинок")],
                            detectPeople: bool = Query(
                                True, 
                                description="Флаг нужно ли обнаруживать людей"
                            ),
                            detectCans: bool = Query(
                                True, 
                                description="Флаг нужно ли обнаруживать котнейнеры, места для сбора мусора, объявления"
                            ),
                            detectGraffiti: bool = Query(
                                True, 
                                description="Флаг нужно ли обнаруживать граффити"
                            ),
                            detectGarbage: bool = Query(
                                True, 
                                description="Флаг нужно ли обнаруживать мусор"
                            ),
                        ):
    """
        Принимает на вход фотографии в формате, поддерживающемся библиотекой cv2, архивы формата zip внутри которых содержатся фотографии подобного типа.
        
        Функция обнаруживает на переданных изображениях такие объекты как: 
        
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
       
        Каждый из параметров detectPeople, detectCans, detectGraffiti и detectGarbage 
        может быть использован для включения или отключения поиска соответствующих объектов в предоставленных изображениях.
        
        ### Аргументы:
        
        * **files** (List[UploadFile]): список загружаемых файлов изображений для анализа.
        * **detectCans** (bool) = True: если True, будет пытаться обнаружить следующие объекты: контейнеры, площадки, объявления.
        * **detectGraffiti** (bool) = True: если True, будет пытаться обнаружить граффити.
        * **detectGarbage** (bool) = True: если True, будет пытаться обнаружить мусор.
        * **detectPeople** (bool) = True: если True, будет пытаться обнаружить людей.
        
        ### Возвращаемое значение:
        
        Каждая детекция — это словарь вида {'название объекта':[список BBox'ов]}.
        Для каждой детекции возвращает ограничивающие рамки объекта.   
        
        {result: {'object_name':[{box_points}]}}.
    """
    batch = await files_to_batch(files)
    return  {"result": get_all_detection_by_batch(batch, detectPeople, detectCans, detectGraffiti, detectGarbage,) }
    
@router.post("/detect-objects-and-classify-сontainerStatus-containerFilling-garbageType", tags=["basic"])
async def detect_objects_and_classify_сontainerStatus_containerFilling_garbageType(files: Annotated[List[UploadFile], File(description="Батч картинок")],
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
                            сontainerStatus: bool = Query(
                                True, 
                                description="Флаг нужно ли определять повреждены ли контейнеры"
                            ),
                        ):
    """
    Принимает на вход фотографии в формате, поддерживающемся библиотекой cv2, архивы формата zip, внутри которых содержатся фотографии подобного типа.
    
    Функция обнаруживает на переданных изображениях такие объекты как: 
    
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
    
    * **files** (List[UploadFile]): список загружаемых файлов изображений для анализа.
    * **detectCans** (bool) = True: если True, будет пытаться обнаружить следующие объекты: контейнеры, площадки, объявления.
    * **detectGraffiti** (bool) = True: если True, будет пытаться обнаружить граффити.
    * **detectGarbage** (bool) = True: если True, будет пытаться обнаружить мусор.
    * **detectPeople** (bool) = True: если True, будет пытаться обнаружить людей.
    * **garbageType** (bool) = True: если True, будет определять тип мусора.
    * **containerFilling** (bool) = True: если True, будет определять заполненность контейнеров.
    * **сontainerStatus** (bool) = True: если True, будет определять состояние контейнеров.
    
    ### Возвращаемое значение:
    
    Для каждой детекции возвращает ограничивающие рамки объекта, а также его класcы если они у него есть. 
    Каждая детекция - это словарь {'название объекта':[список BBox'ов, Название класса: атрибут]}
    """
    batch = await files_to_batch(files, return_fnames=False)
    result = get_all_detection_by_batch(batch, detectPeople, detectCans, detectGraffiti, detectGarbage, save_mask=True)
    add_attributes(batch, result, garbageType, containerFilling, сontainerStatus)
    return  {"result": result } 