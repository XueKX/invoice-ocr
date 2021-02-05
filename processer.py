# -*- coding: utf-8 -*-
'''
@author: Carry
@file: processer.py
@time: 2020/10/27 19:04 
@desc: 
'''
import base64
import functools
import logging
import time

import cv2
import numpy as np

template_boxs = []
name = ['buyer_name', 'buyer_id', 'buyer_address_phone', 'buyer_bank_no', 'seller_name', 'seller_id',
        'seller_address_phone', 'seller_bank_no', 'bill_no', 'price', 'tax', 'account_cap', 'account_lower', 'date']

buyer_name = [375, 272, 1092, 272, 1092, 316, 375, 316]
buyer_id = [375, 316, 1092, 316, 1092, 360, 375, 360]
buyer_address_phone = [375, 360, 1092, 360, 1092, 407, 375, 407]  # nation=[245,105,308,105,308,135,245,135]
buyer_bank_no = [375, 407, 1092, 407, 1092, 453, 375, 453]  # birthday=[116,156,335,156,335,184,116,184]
seller_name = [375, 922, 1092, 922, 1092, 966, 375, 966]
seller_id = [375, 966, 1092, 966, 1092, 1009, 375, 1009]
seller_address_phone = [375, 1009, 1092, 1009, 1092, 1050, 375, 1050]
seller_bank_no = [375, 1050, 1092, 1050, 1092, 1089, 375, 1089]
bill_no = [1407, 72, 1654, 72, 1654, 143, 1407, 143]
price = [1232, 793, 1489, 793, 1489, 838, 1232, 838]
tax = [1586, 793, 1842, 793, 1842, 838, 1586, 838]
account_cap = [629, 860, 1356, 860, 1356, 908, 629, 908]
account_lower = [1459, 860, 1838, 860, 1838, 910, 1459, 910]
date = [1540, 193, 1848, 193, 1848, 241, 1540, 241]

template_boxs.append(buyer_name)
template_boxs.append(buyer_id)
template_boxs.append(buyer_address_phone)
template_boxs.append(buyer_bank_no)
template_boxs.append(seller_name)
template_boxs.append(seller_id)
template_boxs.append(seller_address_phone)
template_boxs.append(seller_bank_no)
template_boxs.append(bill_no)
template_boxs.append(price)
template_boxs.append(tax)
template_boxs.append(account_cap)
template_boxs.append(account_lower)
template_boxs.append(date)

bill_dict = dict(zip(name, template_boxs))

x_threshold = 10
y_threshold = 10
x_Lengthen = 5
y_lengthen = 5

kp1 = des1 = hf1 = wf1 = None


def call_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info('-' * 20 + ' {} call_time:{} '.format(func.__name__, str(end_time - start_time)) + '-' * 20)
        return result

    return wrapper


def img2base64(imgpath):
    with open(imgpath, "rb") as f:  # 转为二进制格式
        base64_data = base64.b64encode(f.read()).decode()  # 使用base64进行加密
    return base64_data


def img_resize(imggray, dwidth):
    '''
    等比缩放
    :param imggray:
    :param dwidth:
    :return:
    '''
    crop = imggray
    size = crop.get().shape
    height = size[0]
    width = size[1]
    height = height * dwidth / width
    crop = cv2.resize(src=crop, dsize=(dwidth, int(height)), interpolation=cv2.INTER_CUBIC)
    return crop


@call_time
def get_perspective_img(parse_img_name):
    '''
    获取透视变换后的图片
    :return:
    '''
    logging.info('-' * 20 + ' 开始处理： ' + '-' * 20)
    W = 1920  # resize后的宽
    template_img_name = './images/template_bak.jpg'
    # test_img = './carry_zone/test_04.jpg'

    if 1:
        # SIFT 精度高 慢 20s
        surf = cv2.xfeatures2d.SIFT_create()
    else:
        # SURF（加速稳健特征）算法 速度快
        surf = cv2.xfeatures2d.SURF_create(1000)  # 默认100，关键点检测的阈值，越高监测的点越少
    global kp1, des1, hf1, wf1
    if kp1 is None:
        # UMat是一个图像容器
        template_img = img_resize(cv2.UMat(cv2.imread(template_img_name, cv2.IMREAD_GRAYSCALE)), W)
        hf1, wf1 = cv2.UMat.get(template_img).shape

        # 返回keypoints是检测关键点，descriptor是描述符，这是图像一种表示方式，可以比较两个图像的关键点描述符，可作为特征匹配的一种方法。
        kp1, des1 = surf.detectAndCompute(template_img, None)

        # 画出关键点（特征点）
        # kpImgA = cv2.drawKeypoints(imageA, kp1, imageA)
        # # kpImgB = cv2.drawKeypoints(grayB, keypointsB, imageB)
        # cv2.imshow("kpImgA", kpImgA)

    parse_img = img_resize(cv2.UMat(parse_img_name), W)
    # parse_img = img_resize(cv2.UMat(cv2.imread(test_img, cv2.IMREAD_GRAYSCALE)), W)

    kp2, des2 = surf.detectAndCompute(parse_img, None)

    # 用FlannBasedMatcher方法进行特征点匹配,寻找最近邻近似匹配
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=10)
    # 匹配耗时 待优化
    matches = cv2.FlannBasedMatcher(index_params, search_params).knnMatch(des1, des2, k=2)

    # 通过描述符的距离进行选择需要的点
    coff = 0.5  # 0.1  0.2  0.8
    good_matches = [m for m, n in matches if m.distance < coff * n.distance]
    if len(good_matches) < 10:
        return None

    # 获取映射变换矩阵，findHomography 计算多个二维点对之间的最优单映射变换矩阵H
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    m, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    m_r = np.linalg.inv(m)
    result_img = cv2.warpPerspective(parse_img, m_r, (wf1, hf1), borderValue=[255, 255, 255])
    cv2.imwrite("./images/_perspective.jpg", result_img.get())
    result_img = result_img.get().astype(np.uint8)
    return result_img


def make_iou(box1, box2):
    """
    计算两个矩形的交集和并集的比
    :param box1:[x1,y1,x2,y2] 左上角的坐标与右下角的坐标
    :param box2:[x1,y1,x2,y2]
    :return: iou_ratio--交并比
    """
    width1 = abs(box1[2] - box1[0])
    height1 = abs(box1[1] - box1[3])  # 这里y1-y2是因为一般情况y1>y2，为了方便采用绝对值
    width2 = abs(box2[2] - box2[0])
    height2 = abs(box2[1] - box2[3])
    x_max = max(box1[0], box1[2], box2[0], box2[2])
    y_max = max(box1[1], box1[3], box2[1], box2[3])
    x_min = min(box1[0], box1[2], box2[0], box2[2])
    y_min = min(box1[1], box1[3], box2[1], box2[3])
    iou_width = x_min + width1 + width2 - x_max
    iou_height = y_min + height1 + height2 - y_max
    if iou_width <= 0 or iou_height <= 0:
        iou_ratio = 0
    else:
        iou_area = iou_width * iou_height  # 交集的面积
        box1_area = width1 * height1
        box2_area = width2 * height2
        iou_ratio = iou_area / (box1_area + box2_area - iou_area)  # 并集的面积
    return iou_ratio


@call_time
def get_area(detect_boxes, w, h):
    '''
    遍历检测框 获取指定区域
    :param detect_boxes:
    :return:
    '''
    boxes = four_point_2_two(detect_boxes)

    boxes_new = two_point_2_four(boxes)

    new_rectangle_list = merge_boxs(boxes_new, w, h)
    print(len(detect_boxes), len(new_rectangle_list))

    draw_rectangle(new_rectangle_list)

    ret_dic = {}
    for i, key in enumerate(bill_dict.keys()):
        mark_iou_ratio = 0.1
        template_box = bill_dict[key]
        template_box_new = [template_box[0], template_box[1],
                            template_box[4], template_box[5]]
        for detect_boxe in new_rectangle_list:
            detect_boxe_new = detect_boxe
            iou_ratio = make_iou(detect_boxe_new, template_box_new)
            if iou_ratio > mark_iou_ratio:
                mark_iou_ratio = iou_ratio
                print(key, iou_ratio)
                ret_dic[key] = detect_boxe
    return ret_dic


def merge_boxs(point_list, w, h):
    """
    将list中矩形合并，并将新矩形增加高度和宽度返回
    :param point_list: 矩形list
    :param w: 图片宽
    :param h: 图片高
    :return: 新的矩形list
    """

    def two_point_compare(point_1, point_2):
        """
        看两个矩形是否靠近，如果两个矩形的距离在一定范围之内，则生成一个新的矩形，
        新矩形的左上角点为原先两个矩形左边矩形的左上角点
        新矩形的右下角点为原先两个矩形右边边矩形的右下角点
        :param point_1: 矩形1
        :param point_2: 矩形2
        :return: 新矩形
        """
        if point_1[0] >= point_2[0]:
            # point_1 在 point_2 右边
            point_change_1 = point_2
            point_change_2 = point_1
        else:
            point_change_1 = point_1
            point_change_2 = point_2
        if abs(point_change_1[2] - point_change_2[0]) <= x_threshold and abs(
                point_change_1[3] - point_change_2[1]) <= y_threshold:
            if point_change_1[3] - point_change_2[1] > 0:
                new_point = (
                    point_change_1[0], point_change_2[1],
                    point_change_2[2], point_change_2[3],
                    point_change_2[4], point_change_2[5],
                    point_change_1[6], point_change_1[7])
            else:
                new_point = (
                    point_change_1[0], point_change_1[1],
                    point_change_2[2], point_change_2[3],
                    point_change_2[4], point_change_2[5],
                    point_change_1[6], point_change_1[7])
        else:
            new_point = ()

        return new_point

    def get_new_list(old_list):
        """
        将原list中的矩形判断距离，距离在一定范围之内的矩形按照上面的方式合并成一个新的矩形，并将新矩形添加到list中
        ，并将原先两个矩形从list中移除。
        :param old_list: 矩形的list
        :return: 新生成的list
        """
        for i in range(len(old_list)):
            if i == len(old_list):
                break
            point1 = old_list[i]
            for j in range(i + 1, len(old_list)):
                point2 = old_list[j]
                if point1 != point2:
                    new_point = two_point_compare(point1, point2)
                    if new_point != ():
                        old_list.remove(point1)
                        old_list.remove(point2)
                        old_list.append(new_point)
                        break
                    else:
                        pass

        return old_list

    def get_new_point_list(point_list):
        """
        将list中的在一定范围内的矩形都合并
        :param point_list: 矩阵list
        :return: 合并矩阵之后的矩阵
        """
        len_ori = len(point_list)
        while True:
            new_list = get_new_list(point_list)
            len_new = len(new_list)
            if len_ori == len_new:
                break
            len_ori = len(new_list)
        return new_list

    def point_change(point, w, h):
        """
        将矩形增加宽度和高度，但是增加后的矩形的顶点要在图片中
        :param point: 矩形
        :param w: 图片的宽度
        :param h: 图片的高度
        :return: 新矩形
        """
        x1 = point[0]
        y1 = point[1]
        x2 = point[4]
        y2 = point[5]
        return (x1 - x_Lengthen if (x1 - x_Lengthen > 0) else 0,
                y1 - y_lengthen if (y1 - y_lengthen > 0) else 0,
                x2 + x_Lengthen if (x2 + x_Lengthen < w) else w,
                y2 + y_lengthen if (y2 + y_lengthen < h) else h)

    new_list = []
    old_list = get_new_point_list(point_list)
    for old_point in old_list:
        new_point = point_change(old_point, w, h)
        if new_point[0] < new_point[2] and new_point[1] < new_point[3]:
            new_list.append(new_point)

    return new_list


def draw_rectangle(new_rectangle_list):
    '''
    画出矩形
    :param new_rectangle_list:
    :return:
    '''
    img_path = './images/detect_boxs.jpg'
    img = cv2.imread(img_path)
    for detect_boxes in new_rectangle_list:
        # 画矩形,红色的线框出来。
        if len(detect_boxes) == 4:
            cv2.rectangle(img=img, pt1=(int(detect_boxes[0]), int(detect_boxes[1])),
                          pt2=(int(detect_boxes[2]), int(detect_boxes[3])),
                          color=(0, 0, 255), thickness=2)
    cv2.imwrite('./images/merge_boxs.jpg', img)


def change_boxs(detect_boxes):
    new_boxs = []
    for detect_box in detect_boxes:
        new_boxs.append((detect_box[2][0], detect_box[2][1],
                         detect_box[3][0], detect_box[3][1],
                         detect_box[0][0], detect_box[0][1],
                         detect_box[1][0], detect_box[1][1]))
    return new_boxs


def four_point_2_two(point_list):
    '''
    四个乱序坐标转左上和右下两个点坐标
    :param point_list:
    :return:
    '''
    result = []
    for old_point in point_list:
        new_list = []
        x = [int(old_point[0][0]), int(old_point[1][0]), int(old_point[2][0]), int(old_point[3][0])]
        y = [int(old_point[0][1]), int(old_point[1][1]), int(old_point[2][1]), int(old_point[3][1])]
        new_list.append(min(x))
        new_list.append(min(y))
        new_list.append(max(x) + 5)
        new_list.append(max(y) + 1)
        result.append(new_list)
    return result


def two_point_2_four(point_list):
    '''
    左上和右下两个点坐标转四个点坐标
    :param point_list:
    :return:
    '''
    result = []
    for old_point in point_list:
        w = old_point[2] - old_point[0]
        h = old_point[3] - old_point[1]
        new_list = []
        new_list.append(old_point[0])
        new_list.append(old_point[1])
        new_list.append(old_point[0] + w)
        new_list.append(old_point[1])
        new_list.append(old_point[2])
        new_list.append(old_point[3])
        new_list.append(old_point[0])
        new_list.append(old_point[1] + h)
        result.append(new_list)
    return result


def get_cut_image(coordinate_point, image):
    """
    根据四个顶点的坐标，将坐标围成的长方形图片从原图中裁减出来
    :param coordinate_point:四个顶点的坐标
    :param image: 原图
    :return: 裁减出来的图片
    """
    x0 = coordinate_point[0]
    x1 = coordinate_point[2]
    y0 = coordinate_point[1]
    y1 = coordinate_point[3]
    cropped = image[y0:y1, x0:x1]
    return cropped


# def fix_ret(key):
#     if key == ''


if __name__ == '__main__':
    import re

    name = '名，。称：上海团迈贸易有限公司'
    name = '称：上海团迈贸易有限公司'
    name = '名  称：上海团称迈贸易有限公司'
    s = re.sub('名?.*称：', "", name)
    # if ret_dic['buyer_name'].startswith('称：'):
    #     ret_dic['buyer_name'] = ret_dic['buyer_name'].replace('称：', '')
    #     print(ret_dic['buyer_name'])

    print(s)
