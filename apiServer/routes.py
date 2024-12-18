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
    dt = detect.create_detection(id = id, weight=weight, source = source)
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