from gevent.pywsgi import WSGIServer
from app import app
if __name__ == '__main__':
    WSGIServer(('127.0.0.1', 5000), app).serve_forever()

