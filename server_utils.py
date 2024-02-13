import zipfile
from io import BytesIO
from typing import List
import torch

import cv2
import base64
import numpy as np
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from torchvision import transforms
from PIL import Image

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
        image = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        if for_attr:
            image = Image.fromarray(image)
            batch.append(transform_tests(image).unsqueeze(0))
        else:
            batch.append(image)
    if for_attr:
        batch = torch.cat(batch)
    if return_fnames:
        return batch, fnames
    return batch
    
def normilise_batch_cv2(batch_pil):
    #to do empty batch
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


def wrap_video_as_zip(video_path):
    zip_filename = "videos.zip"

    # Create an in-memory buffer to store the zip file
    buffer = BytesIO()

    # Create a zip file in the buffer
    with zipfile.ZipFile(buffer, "w") as zipf:
        # Add the video file to the zip file
        zipf.write(video_path, arcname="video.mp4")

    # Set the buffer position to the beginning
    buffer.seek(0)

    # Return the zip file as a streaming response
    return StreamingResponse(buffer, media_type="application/zip",
                             headers={"Content-Disposition": f"attachment; filename={zip_filename}"})
