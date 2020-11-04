# -*- coding:utf-8 -*-
import base64
import functools
import io
import os
import time

import cv2
from PIL import Image
from flask import Flask, request, render_template, jsonify
import json
import logging
import numpy as np
from logging.handlers import TimedRotatingFileHandler

from werkzeug.utils import secure_filename

from paddleocr.ocr_api import do_ocr
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


@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    '''
    页面展示
    :return:
    '''
    # 设置允许的文件格式
    ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        uuid = request.form.get("uuid")
        fileName = 'upload_images/' + uuid + '_' + secure_filename(f.filename)

        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(basepath, 'static', fileName)
        f.save(upload_path)

        ret = {
            'k1': 'v1',
            'k2': 'v1',
            'k3': 'v1',
            'k4': 'v1',
        }
        return render_template('upload.html', fileName=fileName, ret=ret)
    return render_template('upload.html')


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

    ret_box_dic = get_area(boxes, perspective_img.shape[1], perspective_img.shape[0])

    ret_data = {}
    for i, key in enumerate(ret_box_dic.keys()):
        cut_image = get_cut_image(ret_box_dic[key], perspective_img)
        cv2.imwrite(path + '/{}.jpg'.format(key), cut_image)
        ocr_ret = do_ocr(path + '/{}.jpg'.format(key))
        ret_data[key] = ocr_ret[0][0]

    return json.dumps(ret_data, ensure_ascii=False)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=False)
