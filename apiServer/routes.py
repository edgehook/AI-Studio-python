from flask import Blueprint, jsonify, request

from apiServer.detection import detect
from apiServer.utils import responce

bp = Blueprint('main', __name__)

@bp.route('/detect', methods=["POST"])
def start_detect():
    json = request.get_json()
    weight = json["weight"]
    source = json["source"]
    id = json["id"]
    project = json["project"]
    labels = json["labels"]
    name = json["name"]
    print(f"weight:{weight}, source:{source}, id: {id}, project:{project}, labels:{labels}")
    
    dt = detect.create_detection(weights=weight, source = source, project = project, labels = labels, name = name, detect_id= id)
    dt.start_detect()
    result = responce.result(200, "success")
    return jsonify(result), 200

@bp.route('/detect/stop', methods=["POST"])
def stop_detect():
    json = request.get_json()
    id = json["id"]
    dt = detect.get_detection(id = id)
    if dt is not None:
        dt.stop_detect()
    
    result = responce.result(200, "success")
    return jsonify(result), 200

@bp.route('/detect/result/<detect_id>', methods=["GET"])
def get_detect_result(detect_id):
    dt = detect.get_detection(detect_id)
    msg = ""
    if dt is not None:
        msg = dt.image_base64
    return msg, 200