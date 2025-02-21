import os
import random
import string
from flask import Blueprint, jsonify, request, Flask
from apiServer.detection import detect
from apiServer.utils import responce
from flask_cors import CORS
from apiServer.test import face_attendance

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
def start_detect():
    json = request.get_json()
    weight = json["weight"]
    source = json["source"]
    id = json["id"]
    project = json["project"]
    labels = json["labels"]
    name = json["name"]
    thres = json["thres"]
    print(f"weight:{weight}, source:{source}, id: {id}, project:{project}, labels:{labels}, thres: {thres}")
    detect_region = json["detectPoints"]
    print(detect_region)
    base64Img = detect.get_camera_screen(source)
    if base64Img == "":
        result = responce.result(500, "error", "Camera open error")
        return jsonify(result), 500

    dt = detect.create_detection(weights=weight, source = source, project = project, labels = labels, name = name, detect_id= id, detect_region= detect_region, thres=thres)
    dt.start_detect()
    result = responce.result(200, "success")
    return jsonify(result), 200

@bp.route('/v1/detect/stop', methods=["POST"])
def stop_detect():
    json = request.get_json()
    id = json["id"]
    dt = detect.get_detection(detect_id=id)
    if dt is not None:
        dt.stop_detect()
    
    result = responce.result(200, "success")
    return jsonify(result), 200

@bp.route('/v1/camera/screen', methods=["GET"])
def get_camera_screen():
    source = request.args.get("source")
    base64Img = detect.get_camera_screen(source)
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
    base64Img = detect.get_camera_screen(file_path)
    os.remove(file_path)
    os.rmdir(save_path)
    result = responce.result(200, "success",base64Img)
    return jsonify(result), 200

@bp.route('/v1/detect/face', methods=["POST"])
def face_detect():
    json = request.get_json()
    source = json["source"]
    face_attendance(source) 
    result = responce.result(200, "success")
    return jsonify(result), 200