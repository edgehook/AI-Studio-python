import os
import platform
import sys
import threading
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
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

detection_map = {}
def create_detection(id,  weight, source, thres=0.25, view_img=False, project="../../runs/detect", name="exp"):
    dt= detection(weight= weight, source= source, thres= thres, view_img= view_img, project= project, name= name)
    detection_map[id] = dt
    return dt

def get_detection(id):
    dt = detection_map.get(id, None)
    return dt
 
def is_timestamp_more_than_minutes(timestamp_end, timestamp_start):
    diff = timestamp_end - timestamp_start
    LOGGER.info(diff)
# 检查时间差是否等于10分钟
    return diff > 1*60
class detection:
    def __init__(self, weight, source, thres, view_img, project, name):
        self.thread = None
        self.weight = weight
        self.source = source
        self.conf_thres=thres
        self.view_img= view_img
        self.project= project
        self.name= name 
        self.stop = False
    def start_detect(self):
        if self.thread is None or not self.thread.is_alive():
            print("aaa")
            kwargs = {"weights": self.weight, "source": self.source, "conf_thres": self.conf_thres,"project": self.project, "name":self.name}
            self.thread = threading.Thread(target=self.detect, kwargs=kwargs)
            print("bbb")
            self.thread.start()
            print("ccc")
        else:
            LOGGER.info("Thread is already running.")
    def stop_detect(self):
        self.stop = True

    def detect(
        self,
        weights=ROOT / "../../yolov5s.pt",  # model path or triton URL
        source=ROOT / "../../data/images",  # file/dir/URL/glob/screen/0(webcam)
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
        project=ROOT / "../../runs/detect",  # save results to project/name
        name="exp",  # save results to project/name
        exist_ok=True,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        labels=None,  # filter by labels: --labels 0, or --labels 0 2 3
    ):
        source = str(source)
        save_img = not nosave and not source.endswith(".txt")  # save inference images
        is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        webcam = source.isnumeric() or source.endswith(".streams") or (is_url and not is_file)
        screenshot = source.lower().startswith("screen")
        if is_url and is_file:
            source = check_file(source)  # download

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=True)  # increment run
        # save_dir = Path(project) / name
        save_dir.mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        device = select_device(device)
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size
        if webcam:
            # view_img = check_imshow(warn=True)
            dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
            bs = len(dataset)
        elif screenshot:
            dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        vid_path, vid_writer = [None] * bs, [None] * bs
        # signal.signal(signal.SIGINT, signal_handler_wrapper(vid_writer)) 
        # signal.signal(signal.SIGTERM, signal_handler_wrapper(vid_writer)) 
        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
        standardTime = datetime.now()
        saveVideoFileName = ""
        for path, im, im0s, vid_cap, s in dataset:
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
            for i, det in enumerate(pred):  # per image
                seen += 1
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

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = names[c] if hide_conf else f"{names[c]}"
                        confidence = float(conf)
                        confidence_str = f"{confidence:.2f}"
                        if labels is not None:
                            for l in labels:
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
                        
                        if vid_path[i] == None or is_timestamp_more_than_minutes(int(endTime.timestamp()), int(standardTime.timestamp())):  # new video
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
                            im0 = cv2.resize(im0, (w // 5, h // 5))

                        vid_writer[i].write(im0)
            detectImgPath =  str(save_dir / "detect.jpg")
            cv2.imwrite(detectImgPath, im0)
            # Print time (inference-only)
            LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{saveVideoFileName}")
            if self.stop:
                for item in vid_writer:
                    item.release()
                return

        # Print results
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)
        if update:
            strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)



