# Анализ состояния площадки для сбора мусора - Garbage Recognitio (GR)

В этом репозитории собран код для детекции и классификации состояний площадок для сбора мусора через изображение с камер.

## Описание процесса оценивания состаяния площадки

Процесс оценивания состаяния площадки состоит из следующих этапов:

1. Детекция людей, контейнеров для сбора мусора, места для сбора мусора, граффити, рекламы
2. Сегментация мусора, кроме того что внутри контейнеров
3. Вырезание мусорных контейнеров и мусора для последующей классификации
4. Классификация объектов

### Детекция объектов

Детекция производится с помощью моделей YOLOv8 предварительно обученный на кастомном датасете.

Эта модель принимает на вход картинку и выдает для каждого найденного на картинке объекта BBox'ы и название объекта.

### Сегментация мусора

Сегментация производится с помощью моделей YOLOv8 предварительно обученный на кастомном датасете.
Создается копия исходного изображения после чего на предварительно найденных мусорных контейнерах рисуеются беллые прямоугоньки, что бы модель не сегментировала мусор в них.
после этого на изменненом изображении произволдиться сегментация.

### Вырезание объектов и нормализация

Найденные объекты мусорных контейнеров вырезаются по размеру боксов.
Маски мусора вырезаются пустое пространство заполняется пикселями черного цвета.
Размер вырезанных изображений изменяется на (224, 244) и после они нормализуются.

### Классификация объектов

Контейнеры для сбора мусора классифицируются на имеющие повреждения, перевернутые и имеющие удоволетварительное состояние.
Так же контейнры для мусора классифицируются по степени наполнености мусором: пустые, наполовину полные и полные
Мусор классифицируется на ТКО, КГМ и навал мусора
TKO - мелкий мусор.
KGO - крупногабаритные отходы -  твердые коммунальные отходы, размер которых не позволяет осуществить их складирование в контейнерах.
Bulk - навал мусора -  скопление твердых бытовых отходов и крупногабаритного мусора, по объему, не превышающему 1 м3.

## Файлы и папки репозитория

### Веса и датасеты

Скачать веса можно по ссылке: 
https://drive.google.com/drive/folders/1ozJUmHe4ODd-OKXXdUHFid5iacqEMWkR?usp=sharing

Скачать датасеты можно по ссылке: https://drive.google.com/drive/folders/1r4t4I91HpLBa3S_1x8i400TF3x0Q5KgI
https://universe.roboflow.com/itmo-ognxn/russian-garbage-detection/dataset/
https://universe.roboflow.com/itmo-ognxn/selfie-trash/dataset/
https://universe.roboflow.com/itmo-0kdik/graffiti-wvjbp
Датасеты включают в себя:

1. Разметку мусорный контейнеров, мест для мусора, объявлений
2. Сегментацию мусора
3. Классификацию мусорных контейнеров
4. Классификацию мусора

### Файлы

* [align_utils.py](align_utils.py) - файл с функциями для вырезания и рисования белых прямоугоников на контейнеры

## FastAPI Server

### Структура файлов

* [requirements.txt](requirements.txt) - файл, в котором описаны зависимости, которые необходимо установить. При
  билдинге докер-образа на сервере закомментируйте torch, torchvision и opencv-python, потому что
  в [Dockerfile](Dockerfile) уже прописана их установка.

#### Серверная логика

* [server.py](server.py) - точка входа, в этом файле нужно указать номер порт, по которому сервис будет крутиться на
  сервере
* [server_basic_routers.py](server_basic_routers.py) - базовые методы, которые не делают никакой логики
* [server_test_routers.py](server_test_routers.py) - тестовые методы, которые имеют какую-то логику. Например,
  обрабатывают поток.
* [server_utils.py](server_utils.py) - функции и утилиты для чтения реквестов и записи респонсов.
* [server_models.py](server_models.py) - файл, в котором нужно создать инстансы моделей и других важных параметров
* [client.py](client.py) - файл с функциями, которые проверяют работоспособность базовых методов

#### Ваши файлы и модели

В моем случае:

* [detector_segmentator.py](face_detector.py) - класс для работы детектора и сегментатора.
* [align_utils.py](align_utils.py) - функции для вырезания детекций и сегментаций.

#### Docker

* [docker_build_image.sh](docker_build_image.sh) - скрипт для билда образа. В этом скрипте нужно указать имя
  создаваемого образа.
* [docker_run_container.sh](docker_run_container.sh) - скрипт для запуска контейнера. В этом скрипте нужно точно также
  указать имя образа. Также нужно указать порты.
* [Dockerfile](Dockerfile) - докерфайл, в котором скорее всего ничего менять не придется.
* [.dockerignore](.dockerignore) - файл, в котором нужно указать, какие крупные файлы и папки не нужно использовать при
  билдинге образа (это датасеты, веса, архивы и т.п.). Нужен, чтобы сократить время билдинга образа.
* [entry.sh](entry.sh) - скрипт для запуска сервера, тоже менять ничего не нужно.

#### Остальное
  
  icon - папка с иконками для классификации объектов.
* test* - тестовые файлы, которые можно отправить на вход методам сервера

### Как этим пользоваться

Чтобы запустить сервер локально (не на удаленном сервере), используйте эту команду:

```
uvicorn server:app --reload
```

Это позволит включить дебаг режим и сервер на лету будет подтягивать изменения в файлах и перезапускаться. Т.е. всегда
будет работать актуальная версия.

После запуска сервера перейдите в браузере на `localhost:8000/docs`. Вы увидите автосгенерированную документацию в стиле
Swagger. Здесь можно удобно посылать запросы и тестировать работу методов. 

