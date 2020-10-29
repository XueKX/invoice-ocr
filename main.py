import base64
import functools
import io
import time

from PIL import Image
from flask import Flask, request
import json
import logging
import numpy as np
from logging.handlers import TimedRotatingFileHandler

from processer import get_perspective_img, call_time, get_area
from psenet_tf.eval import detect_pse

log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
formatter = logging.Formatter(log_fmt)
log_file_handler = TimedRotatingFileHandler(filename="app_logs", when="D", interval=1, backupCount=7)
log_file_handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()
log.addHandler(log_file_handler)
app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/ocr', methods=['POST'])
@call_time
def ocr():
    data_request = request.json
    img_name = data_request['name']
    img_code = data_request['image_base64_string']

    image_data = base64.b64decode(img_code)
    img_array = io.BytesIO(image_data)
    image = np.array(Image.open(img_array).convert('L'))

    img_perspective = get_perspective_img(image)

    boxes = detect_pse(img_perspective)

    get_area(boxes)
    return 'ss'


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=False)
