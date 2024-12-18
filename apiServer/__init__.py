from flask import Flask
import os
from . import routes
def run_api_server():
    port = 8090
    if os.getenv('PORT') is not None:
        port = int(os.getenv('PORT'))
    
    app = Flask(__name__)
    app.config["DEBUG"] = True
    app.register_blueprint(routes.bp)
    app.run(host='0.0.0.0',port=port, debug=True)
