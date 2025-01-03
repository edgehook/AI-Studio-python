
import os
from . import routes
from apiServer.transport.websocket import run_server

def run_api_server():
    run_server()
    port = 8090
    if os.getenv('PORT') is not None:
        port = int(os.getenv('PORT'))
    routes.app.register_blueprint(routes.bp)
    routes.app.run(host='0.0.0.0',port=port)
