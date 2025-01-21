import json
import os
import platform
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import base64
import subprocess
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams, gstreamer_pipeline
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    cv2,
    increment_path,
    non_max_suppression,
    scale_boxes,
    strip_optimizer,
)
from utils.torch_utils import select_device, smart_inference_mode
from datetime import datetime, timedelta

from apiServer.transport import notify, websocket


detection_map = {}
def create_detection(weights, source, labels, detect_id, thres=0.25, view_img=False, detect_region=None, project="../../runs/detect", name="exp"):
    dt= detection(
        weights= weights, 
        source= source, 
        thres= thres, 
        view_img= view_img, 
        project= project, 
        name= name,
        labels=labels, 
        detect_id= detect_id,
        detect_region= detect_region)
    detection_map[detect_id] = dt
    return dt

def get_detection(detect_id):
    dt = detection_map.get(detect_id, None)
    return dt
 
def is_timestamp_more_than_minutes(timestamp_end, timestamp_start, interval):
    diff = timestamp_end - timestamp_start
    return diff > interval

def get_camera_screen(source):
    print(f"source: {source}")
    s = eval(source) if source.isnumeric() else source  # i.e. s = '0' local webcam
    if type(s) == int:
        print('[gstreamer] ', gstreamer_pipeline(sensor_id=s, capture_width=1920, capture_height=1080))
        cap = cv2.VideoCapture(gstreamer_pipeline(sensor_id=s, capture_width=1920, capture_height=1080), cv2.CAP_GSTREAMER)
    else:
        print('[gstreamer] ', gstreamer_pipeline())
        cap = cv2.VideoCapture(s)
    if cap.isOpened():
        success, im = cap.read()
        cap.release()
        if success:
            h, w, _ = im.shape
            im0 = cv2.resize(im, (w // 5, h // 5))
            success, encoded_image = cv2.imencode('.jpg', im0)
            if success:
                image_base64 =base64.b64encode(encoded_image.tobytes()).decode('utf-8')
                return image_base64
    return ""

class detection:
    def __init__(self, weights, source, thres, view_img, project, name, labels, detect_id, detect_region):
        self.thread = None
        self.weights = weights
        self.source = source
        self.conf_thres=thres
        self.view_img= view_img
        self.project= project
        self.name= name 
        self.labels = labels
        self.detect_stop = False
        self.is_report = True
        self.image_base64 = ""
        self.detect_id = detect_id
        self.detect_region = detect_region
        self.is_detect = True
    def start_detect(self):
        if self.thread is None or not self.thread.is_alive():
            kwargs = { "conf_thres": self.conf_thres}
            self.thread = threading.Thread(target=self.detect, kwargs=kwargs)
            self.thread.start()
            
        else:
            LOGGER.info("Thread is already running.")
    def stop_detect(self):
        self.detect_stop = True

    def detect(
        self,
        data=ROOT / "../../data/coco128.yaml",  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results

        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
    ):
        
        self.source = str(self.source).lstrip()
        save_img = not nosave and not self.source.endswith(".txt")  # save inference images
        is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = self.source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        webcam = self.source.isnumeric() or self.source.endswith(".streams") or (is_url and not is_file)
        screenshot = self.source.lower().startswith("screen")
        if is_url and is_file:
            self.source = check_file(self.source)  # download

        # Directories
        save_dir = increment_path(Path(self.project) / self.name, exist_ok=True)  # increment run
        # save_dir = Path(self.project) / self.name
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(self.weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            # view_img = check_imshow(warn=True)
            dataset = LoadStreams(self.source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(self.source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(self.source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs
        # signal.signal(signal.SIGINT, signal_handler_wrapper(vid_writer)) 
        # signal.signal(signal.SIGTERM, signal_handler_wrapper(vid_writer)) 
        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
        standardTime = datetime.now()
        detect_start_time = datetime.now()
        hearbeat_start_time = datetime.now()
        saveVideoFileName = ""
        for path, im, im0s, vid_cap, s in dataset:
            #detect_region: [[x,y], [x1,y1], [x2,y2], [x3,y3]]
            if self.is_detect and self.weights:
                if self.detect_region and len(self.detect_region) >0:
                    w = self.detect_region[0][0]
                    h = self.detect_region[0][1]
                    # wl1 = 4 / w # Upper left width ratio
                    # hl1 = 3 / 10 # Upper left height ratio
                    # wl2 = 9.8 / 10  # Upper right width ratio
                    # hl2 = 3 / 10  # Upper right height ratio
                    # wl3 = 9.8 / 10  # Bottom right width ratio
                    # hl3 = 8 / 10  # Bottom right height ratio
                    # wl4 = 4 / 10  # Bottom left width ratio
                    # hl4 = 8 / 10  # Bottom left height ratio
                    if webcam:
                        for b in range(0,im.shape[0]):
                            np_array=[]
                            for item in self.detect_region[1:]:
                                wl = item[0] / w
                                hl = item[1] / h
                                np_array.append([int(im[b].shape[2] * wl), int(im[b].shape[1] * hl)])
                            mask = np.zeros([im[b].shape[1], im[b].shape[2]], dtype=np.uint8)
                            pts = np.array(np_array, np.int32)
                            mask = cv2.fillPoly(mask,[pts],(255,255,255))
                            imgc = im[b].transpose((1, 2, 0))
                            imgc = cv2.add(imgc, np.zeros(np.shape(imgc), dtype=np.uint8), mask=mask)
                            im[b] = imgc.transpose((2, 0, 1))
                    else:
                        np_array=[]
                        for item in self.detect_region[1:]:
                            wl = item[0] / w
                            hl = item[1] / h
                            np_array.append([int(im.shape[2] * wl), int(im.shape[1] * hl)])
                        mask = np.zeros([im.shape[1], im.shape[2]], dtype=np.uint8)
                        pts = np.array(np_array, np.int32)
                        mask = cv2.fillPoly(mask, [pts], (255,255,255))
                        im = im.transpose((1, 2, 0))
                        im = cv2.add(im, np.zeros(np.shape(im), dtype=np.uint8), mask=mask)
                        im = im.transpose((2, 0, 1))
                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim
                    if model.xml and im.shape[0] > 1:
                        ims = torch.chunk(im, im.shape[0], 0)

                # Inference
                with dt[1]:
                    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                    if model.xml and im.shape[0] > 1:
                        pred = None
                        for image in ims:
                            if pred is None:
                                pred = model(image, augment=augment, visualize=visualize).unsqueeze(0)
                            else:
                                pred = torch.cat((pred, model(image, augment=augment, visualize=visualize).unsqueeze(0)), dim=0)
                        pred = [pred, None]
                    else:
                        pred = model(im, augment=augment, visualize=visualize)
                # NMS
                with dt[2]:
                    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)      
                # Process predictions
                for i, det in enumerate(pred): 
                    notify_data = []
                    # per image
                    seen += 1
                    if self.detect_region and len(self.detect_region) >0:
                        w = self.detect_region[0][0]
                        h = self.detect_region[0][1]
                        np_array = []
                        if webcam:  # batch_size >= 1
                            p, s, im0, frame = path[i], f'{i}: ', im0s[i].copy(), dataset.count
                            for item in self.detect_region[1:]:
                                wl = item[0] / w
                                hl = item[1] / h
                                np_array.append([int(im0.shape[1] * wl), int(im0.shape[0] * hl)])
                            # print(np_array)
                            cv2.putText(im0, "Detection Region",(int(np_array[0][0]), int(np_array[0][1]) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

                            pts = np.array(np_array, np.int32)  # pts4
                            zeros = np.zeros((im0.shape), dtype=np.uint8)
                            mask = cv2.fillPoly(zeros, [pts], color=(255,255,255))
                            im0 = cv2.addWeighted(im0, 1, mask, 0.2, 0)
                            #cv2.polylines(im0, [pts], True, (0, 0, 255), 3)
                        else:
                            p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)
                            cv2.putText(im0, "Detection Region", (int(np_array[0][0]), int(np_array[0][1] - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                            pts = np.array(np_array, np.int32)  # pts4
                            # pts = pts.reshape((-1, 1, 2))
                            zeros = np.zeros((im0.shape), dtype=np.uint8)
                            mask = cv2.fillPoly(zeros, [pts], color=(255,255,255))
                            im0 = cv2.addWeighted(im0, 1, mask, 0.2, 0)
                            #cv2.polylines(im0, [pts], True, (255, 255, 0), 3)  
                    else:

                        if webcam:  # batch_size >= 1
                            p, im0, frame = path[i], im0s[i].copy(), dataset.count
                            s += f"{i}: "
                        else:
                            p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # im.jpg
                    txt_path = str(save_dir / "labels" / p.stem) + ("" if dataset.mode == "image" else f"_{frame}")  # im.txt
                    s += "%gx%g " % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            if self.labels is not None and isinstance(self.labels, list) and len(self.labels) >0:
                                for l in self.labels:
                                    if names[int(c)] == l:
                                        notify_data.append({"label": names[int(c)], "count": f"{n}"})
                            else:
                                notify_data.append({"label": names[int(c)], "count": f"{n}"})
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            c = int(cls)  # integer class
                            label = names[c] if hide_conf else f"{names[c]}"
                            confidence = float(conf)
                            confidence_str = f"{confidence:.2f}"
                            if self.labels is not None and isinstance(self.labels, list) and len(self.labels) >0:
                                for l in self.labels:
                                    # LOGGER.info("label:"+l+"## detect name:"+names[int(cls)])
                                    if l == label:
                                        if save_img or view_img:  # Add bbox to image
                                            c = int(cls)  # integer class
                                            label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                                            annotator.box_label(xyxy, label, color=colors(c, True))


                            else:
                                if save_img  or view_img:  # Add bbox to image
                                    c = int(cls)  # integer class
                                    label = None if hide_labels else (names[c] if hide_conf else f"{names[c]} {conf:.2f}")
                                    annotator.box_label(xyxy, label, color=colors(c, True))
                    # Stream results
                    im0 = annotator.result()
                    if view_img:
                        if platform.system() == "Linux" and p not in windows:
                            windows.append(p)
                            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                        cv2.imshow(str(p), im0)
                        cv2.waitKey(1)  # 1 millisecond

                    # Save results (image with detections)
                    if save_img:
                        if dataset.mode == "image":
                            cv2.imwrite(save_path, im0)
                        if dataset.mode == "video":
                            if vid_path[i] != save_path:  # new video
                                vid_path[i] = save_path
                                if isinstance(vid_writer[i], cv2.VideoWriter):
                                    vid_writer[i].release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path = str(Path(save_path).with_suffix(".mp4"))  # force *.mp4 suffix on results videos
                                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                            vid_writer[i].write(im0)
                        else:  # 'stream'
                            endTime = datetime.now()
                            
                            if vid_path[i] == None or is_timestamp_more_than_minutes(int(endTime.timestamp()), int(standardTime.timestamp()), 60*1):  # new video
                                standardTime = endTime
                                time_str = endTime.strftime("%y-%m-%d-%H-%M-%S")
                                save_path = str(save_dir / time_str)
                                save_path = str(Path(save_path).with_suffix(".mp4")) 
                                saveVideoFileName = str(Path(time_str).with_suffix(".mp4")) 
                                vid_path[i] = save_path
                                if isinstance(vid_writer[i], cv2.VideoWriter):
                                    vid_writer[i].release()  # release previous video writer
                                if vid_cap:  # video
                                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                else:  # stream
                                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                                # force *.mp4 suffix on results videos
                                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                                im0 = cv2.resize(im0, (w // 10, h // 10))

                            vid_writer[i].write(im0)
                detectImgPath =  str(save_dir / "detect.jpg")
                cv2.imwrite(detectImgPath, im0)
                # print(websocket.websocket_map)
                websocket_connections = websocket.websocket_map.get(self.detect_id, None) 
                if websocket_connections:
                    success, encoded_image = cv2.imencode('.jpg', im0)
                    if success:
                        image_base64 =base64.b64encode(encoded_image.tobytes()).decode('utf-8')
                        websocket.push_msg(self.detect_id, image_base64)
                    
                # Print time (inference-only)
                LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{saveVideoFileName}")
                detect_report_time = datetime.now()
                # push nofity msg
                if len(notify_data):
                    if self.is_report:
                        success, encoded_image = cv2.imencode('.jpg', im0)
                        image_base64 = ""
                        if success:
                            image_base64 =base64.b64encode(encoded_image.tobytes()).decode('utf-8')
                        notify.push_notify_msg({"type": "object_detect", "data": json.dumps(notify_data), "image": image_base64, "id": self.detect_id, "name": self.name, "videoFileName": saveVideoFileName})
                        self.is_report = False
                    else:
                        if is_timestamp_more_than_minutes(int(detect_report_time.timestamp()), int(detect_start_time.timestamp()), 60*10):
                            detect_start_time = detect_report_time
                            self.is_report = True
               
            else:
                h, w, _ = im0s[0].shape
                im0 = cv2.resize(im0s[0], (w // 5, h // 5))
                success, encoded_image = cv2.imencode('.jpg', im0)
                if success:
                    image_base64 =base64.b64encode(encoded_image.tobytes()).decode('utf-8')
                    websocket.push_msg(self.detect_id, image_base64)
            # push hearbeat
            hearbeat_report_time = datetime.now()
            if is_timestamp_more_than_minutes(int(hearbeat_report_time.timestamp()), int(hearbeat_start_time.timestamp()), 30):
                hearbeat_start_time = hearbeat_report_time
                notify.push_hearbeat_msg(self.detect_id) 
            if self.detect_stop:
                if len(vid_writer) >0:
                    for item in vid_writer:
                        item.release()
                if vid_cap:
                    vid_cap.release()
                dataset.detect_stop = True
                del detection_map[self.detect_id]
                return 
        # Print results
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
        if update:
            strip_optimizer(self.weights[0])  # update model (to fix SourceChangeWarning)



