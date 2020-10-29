import base64
import functools
import io
import os
import time

import cv2
from PIL import Image
from flask import Flask, request
import json
import logging
import numpy as np
from logging.handlers import TimedRotatingFileHandler

from processer import get_perspective_img, call_time, get_area, get_cut_image
from psenet_tf.eval import detect_pse

log_fmt = '%(asctime)s\tFile \"%(filename)s\",line %(lineno)s\t%(levelname)s: %(message)s'
formatter = logging.Formatter(log_fmt)
log_file_handler = TimedRotatingFileHandler(filename="app_logs", when="D", interval=1, backupCount=7)
log_file_handler.setFormatter(formatter)
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()
log.addHandler(log_file_handler)
app = Flask(__name__)

path = './images/result'
if not os.path.exists(path):
    os.makedirs(path)


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

    perspective_img = get_perspective_img(image)

    boxes = detect_pse(perspective_img)

    ret_dic = get_area(boxes, perspective_img.shape[1], perspective_img.shape[0])

    for i, key in enumerate(ret_dic.keys()):
        cut_image = get_cut_image(ret_dic[key], perspective_img)
        cv2.imwrite(path + '/{}.jpg'.format(key), cut_image)

    return 'ss'


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=False)
