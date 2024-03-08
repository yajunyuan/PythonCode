from flask import Flask
from flask import request, jsonify
from flask_cors import CORS
import requests
app = Flask(__name__)
CORS(app)
@app.route("/111", methods=["GET","POST"])
def predict():
    # result = {"success": False}
    return 'Hello World,nihao'

@app.route("/", methods=["GET","POST"])
def get_request():
    """ request数据 """
    params = request.json if request.method == "POST" else request.args
    try:
        data = request.get_json()
        print(data)
    except:
        pass

if __name__ == '__main__':
    app.run(host='192.168.13.21', port=8080, debug=False)