import os
import random
import string
import sys
from flask import Blueprint, jsonify, request, Flask
import torch
from apiServer.detection import detect
from apiServer.utils import jetson_libraries
from apiServer.utils import responce
from flask_cors import CORS
bp = Blueprint('main', __name__)
app = Flask(__name__)
cors = CORS(app, resources={
    "/v1/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
@bp.route('/v1/detect', methods=["POST"])
def run_detect():
    json = request.get_json()
    weight = json["weight"]
    source = json["source"]
    id = json["id"]
    project = json["project"]
    labels = json["labels"]
    name = json["name"]
    thres = json["thres"]
    tp= json["tp"]
    duration = json["duration"]
    isDetect = json["isDetect"]
    print(f"weight:{weight}, source:{source}, id: {id}, project:{project}, labels:{labels}, thres: {thres}, type: {tp}, duration: {duration}, isDetect: {isDetect}")
    detect_region = json["detectPoints"]
    if not detect.camera_is_available(source=source, tp=tp):
        result = responce.result(500, "error", "Camera open error")
        return jsonify(result), 500

    dt = detect.create_detection(
        weights=weight, 
        source = source, 
        project = project, 
        labels = labels, 
        name = name, 
        detect_id= id, 
        detect_region= detect_region, 
        thres=thres, 
        isDetect=isDetect)
    dt.start_detect(tp, duration=duration)
    result = responce.result(200, "success")
    return jsonify(result), 200

@bp.route('/v1/detect/stop', methods=["POST"])
def stop_detect():
    json = request.get_json()
    id = json["id"]
    dt = detect.get_detection(detect_id=id)
    if dt is not None:
        dt.stop_detect()
    else:
        print(f"detect is None, id={id}")
    
    result = responce.result(200, "success")
    return jsonify(result), 200
@bp.route('/v1/detect/monitor', methods=["POST"])
def get_detect_monitor():
    json = request.get_json()
    detectIds = json["detectIds"]
    
    monitor_array = []
    if detectIds is not None:
            # detectIds = json.loads(ids)
        for detectId in detectIds:
            monitor_map = {}
            dt = detect.get_detection(detect_id=detectId)
            if dt is not None:
                detect_time = dt.detect_time
                if not detect_time:
                    detect_time = 0
                handle_time = dt.handle_time
                monitor_map["detectId"] = detectId
                monitor_map["detectTime"] = round(float(detect_time)*1000, 2)
                monitor_map["handleTime"] = round(float(handle_time)*1000, 2)
                monitor_map["fpsTime"]= round(1/float(handle_time), 2)
                monitor_map["name"] = dt.name
                monitor_array.append(monitor_map)
    result = responce.result(200, "success",monitor_array)
    return jsonify(result), 200
@bp.route('/v1/detect/cuda', methods=["GET"])
def get_cuda_version():
    cuda_version = ""
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
    else:
        print("No available CUDA device was detected。")
    result = responce.result(200, "success",cuda_version)
    return jsonify(result), 200

@bp.route('/v1/detect/env', methods=["GET"])
def get_detect_env():
    cuda_version = ""
    if torch.cuda.is_available():
        cuda_version = torch.version.cuda
    else:
        print("No available CUDA device was detected。")
    libs = jetson_libraries.get_libraries()
    print(f"jetson_libraries, {libs}")
    tensorrt_version=""
    if "tensorrt" in libs:
        tensorrt_version = libs["tensorrt"]
    cudnn_version=""
    if "cudnn" in libs:
        cudnn_version = libs["cudnn"]
    vpi_version=""
    if "vpi" in libs:
        vpi_version = libs["vpi"]
    vulkan_version=""
    if "vulkan" in libs:
        vulkan_version = libs["vulkan"]
    python_major_version = sys.version_info.major
    python_minor_version = sys.version_info.minor

    # 组合成 x.y 格式
    python_version = f"{python_major_version}.{python_minor_version}"
    env_map = {}
    env_map["cuda_version"] = cuda_version
    env_map["cudnn_version"] = cudnn_version
    env_map["python_version"] = python_version
    env_map["vpi"] = vpi_version
    env_map["tensorrt_version"] = tensorrt_version
    env_map["vulkan_version"]=vulkan_version
    result = responce.result(200, "success",env_map)
    return jsonify(result), 200
@bp.route('/v1/camera/screen', methods=["GET"])
def get_camera_screen():
    source = request.args.get("source")
    tp = request.args.get("type")
    base64Img = detect.get_camera_screen(source, tp= tp)
    result = responce.result(200, "success", base64Img)
    return jsonify(result), 200

@bp.route('/v1/camera/video/screen', methods=["POST"])
def get_camera_video_screen():
    if 'video' not in request.files:
        return "No file part", 400

    file = request.files['video']

    if file.filename == '':
        return "No selected file", 400

    random_dir = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(10))
    save_path = os.path.join(os.getcwd(), random_dir)
    os.makedirs(save_path, exist_ok=True)

    file_path = os.path.join(save_path, file.filename)
    file.save(file_path)
    base64Img = detect.get_camera_screen(file_path, tp="video")
    os.remove(file_path)
    os.rmdir(save_path)
    result = responce.result(200, "success",base64Img)
    return jsonify(result), 200







