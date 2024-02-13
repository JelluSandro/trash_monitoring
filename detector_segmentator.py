from io import BytesIO
from typing import List

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import cv2

class DetectorSegmentator:
    def __init__(self, device, weight_path, cls_filter=None, fp16=False):
        self.device = device
        self.detector_model = YOLO(weight_path)
        self.detector_model.to(device=self.device)
        self.dict_names = self.detector_model.names
        self.FP16 = fp16
        self.predict_args = {
            'show': False,
            'save': False,
            'conf': 0.4,
            'save_txt': False,
            'save_crop': False,
            'verbose': False,
            'device': self.device,
            'half': self.FP16
        }
        if cls_filter is not None:
            self.dict_names = {k: v for k, v in self.dict_names.items() if k in cls_filter}
            self.predict_args['classes'] = cls_filter
   
    def run(self, batch, save_mask=False):
        self.predict_args['source'] = batch
        results = self.detector_model.predict(**self.predict_args)
        out_data = []
        names = self.dict_names.values()
        
        for res in results:
            res_dict = {name: [] for name in names}
            out_data.append(res_dict)
            boxes = res.boxes
            cls = boxes.cls
            for ind, points in enumerate(boxes.xyxy):
                box_points = points.cpu().numpy().astype(int)
                data_entry = {"box_points": box_points.tolist()}
                if save_mask is True and res.masks is not None:
                    data_entry["mask"] = res.masks[ind].data[0].numpy()
                out_data[-1][self.dict_names[int(cls[ind])]].append(data_entry)
                
        return out_data

class PersonDetector(DetectorSegmentator):
    def __init__(self, device, fp16=False):
        super().__init__(device, 'yolov8n.pt', [0], fp16=fp16)
        
class GraffitiDetector(DetectorSegmentator):
    def __init__(self, device, fp16=False):
        super().__init__(device, './weights/yolov8m-graffiti.pt', fp16=fp16)
        
class CansDetector(DetectorSegmentator):
    def __init__(self, device, fp16=False):
        super().__init__(device, './weights/yolov8m-cans.pt', fp16=fp16)
        
class GarbageSegmentator(DetectorSegmentator):
    def __init__(self, device, fp16=False):
        super().__init__(device, './weights/yolov8seg-garbage.pt', fp16=fp16)

def add_detections_segmentations_on_images(batch: List[np.ndarray],
                                            infos: List,
                                            drawing_callback,
                                            return_bytes=True):
    assert len(batch) == len(infos)

    for (image, info) in tqdm(zip(batch, infos)):
        for key in info:
            for res in info[key]:
                image = drawing_callback(image, res, key)
        if return_bytes:
            _, buffer = cv2.imencode('.png', image)
            image_bytes = buffer.tobytes()
            yield image_bytes
        else:
            yield image