import zipfile
from io import BytesIO
from typing import List
import torch

import cv2
import base64
import numpy as np
from fastapi import UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from torchvision import transforms
from PIL import Image
import os


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

transform_tests = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])

async def files_to_batch(files: List[UploadFile], return_fnames=False, for_attr=False):
    all_files = []
    fnames = []
    for file in files:
        cur_fname = file.filename
        if cur_fname.endswith(".zip"):
            contents = await file.read()
            zip_file = zipfile.ZipFile(BytesIO(contents))
            for file_name in zip_file.namelist():
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_contents = zip_file.read(file_name)
                    all_files.append(file_contents)
                    fnames.append(file_name)
        else:
            file_contents = await file.read()
            all_files.append(file_contents)
            fnames.append(cur_fname)

    batch = []
    for image_bytes in all_files:
        try:
            image = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            #if type(image) != NoneType:
            if for_attr:
                image = Image.fromarray(image)
                batch.append(transform_tests(image).unsqueeze(0))
            else:
                batch.append(image)
        except:
            continue
    if len(batch) == 0:
        raise HTTPException(status_code=404, detail="Images not found")
    if for_attr:
        batch = torch.cat(batch)
    if return_fnames:
        return batch, fnames
    return batch
    
def normilise_batch_cv2(batch_pil):
    batch = []
    for img_pil in batch_pil:
        batch.append(transform_tests(img_pil).unsqueeze(0))
    return torch.cat(batch)

async def transform_batch(files):
    batch = []
    for b64img in files:
        im_bytes = base64.b64decode(b64img)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        pill_img = Image.fromarray(img)
        batch.append(transform_tests(pill_img).unsqueeze(0))
    return torch.cat(batch)

def wrap_images_as_zip(results, fnames=None):
    """
    :param results: list/generator of bytes images
    :param fnames: list of out filenames
    :return: ZIP response
    """
    zip_data = BytesIO()
    with zipfile.ZipFile(zip_data, "w") as zip_file:
        for i, image_bytes in enumerate(results):
            out_name = fnames[i] if fnames is not None else f"image{i + 1}.png"
            zip_file.writestr(out_name, image_bytes)
    zip_data.seek(0)

    return StreamingResponse(
        zip_data, media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=images.zip"}
    )


def wrap_video_as_zip(video_path: str) -> StreamingResponse:
    """
    Wraps a video file specified by video_path into a zip archive
    and returns it as a FastAPI StreamingResponse.

    Parameters:
    - video_path: A path to a video file to be zipped.

    Returns: StreamingResponse with a zip archive containing the video file.
    """
    zip_filename = "videos.zip"
    buffer = BytesIO()

    try:
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Use basename in case video_path is a full path
            video_basename = os.path.basename(video_path)
            zip_file.write(video_path, arcname=video_basename)

        buffer.seek(0)

        return StreamingResponse(
            buffer, 
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename={zip_filename}"
            }
        )

    except FileNotFoundError:
        # Handle the error when the video file is not found.
        # Depending on how you want to structure your API, you might want to raise an HTTPException
        # from fastapi import HTTPException
        # raise HTTPException(status_code=404, detail="Video not found")
        raise HTTPException(status_code=404, detail="Error: The file was not found.")
        # assuming we want to continue in a console-based script

    except zipfile.BadZipFile:
        # Handle other potential zip file errors.
        raise HTTPException(status_code=404, detail="Error: Failed to create a zip file.")
