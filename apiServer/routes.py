from flask import Blueprint, jsonify, request, Flask
from apiServer.detection import detect
from apiServer.utils import responce
from apiServer.transport import websocket
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

@bp.route('/v1/detect/result/<detect_id>', methods=["GET"])
def get_detect_result(detect_id):
    dt = detect.get_detection(detect_id)
    msg = ""
    if dt is not None:
        msg = dt.image_base64
    return msg, 200

@bp.route('/v1/camera/screen', methods=["GET"])
def get_camera_screen():
    source = request.args.get("source")
    base64Img = detect.get_camera_screen(source)
    print(base64Img)
    result = responce.result(200, "success", base64Img)
    return jsonify(result), 200