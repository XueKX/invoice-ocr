# -*- coding: utf-8 -*-
'''
@author: Carry
@file: processer.py
@time: 2020/10/27 19:04 
@desc: 
'''
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
    W = 1920  # resize后的宽
    template_img_name = './images/template.jpg'
    # test_img = './carry_zone/test_04.jpg'

    # SURF（加速稳健特征）算法
    min_hessian = 500
    surf = cv2.xfeatures2d.SURF_create(min_hessian)  # 默认100，关键点检测的阈值，越高监测的点越少
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


if __name__ == '__main__':
    detect_boxes = [
        [[1223.1549072265625, 132.33447265625], [724.8194580078125, 118.58719635009766],
         [726.5159912109375, 56.238067626953125], [1224.851318359375, 69.98534393310547]],
        [[1400.5263671875, 144.67578125], [1351.5789794921875, 144.67578125], [1351.5789794921875, 101.75],
         [1400.5263671875, 101.75]],
        [[309.693115234375, 156.5589599609375], [308.0234069824219, 114.52715301513672],
         [604.5173950195312, 102.5855484008789], [606.1871337890625, 144.6173553466797]],
        [[1690.8558349609375, 156.38619995117188], [1438.5703125, 146.97779846191406],
         [1440.257568359375, 101.11189270019531], [1692.54296875, 110.52029418945312]],
        [[1705.262939453125, 151.03514099121094], [1705.262939453125, 127.18749237060547],
         [1836.315673828125, 127.18749237060547], [1836.315673828125, 151.03514099121094]],
        [[1835.9935302734375, 196.33094787597656], [1679.7852783203125, 189.1815948486328],
         [1681.1298828125, 159.39743041992188], [1837.338134765625, 166.54676818847656]],
        [[1558.3287353515625, 241.7394561767578], [1413.425048828125, 233.63368225097656],
         [1414.9844970703125, 205.36868286132812], [1559.88818359375, 213.47447204589844]],
        [[1811.0526123046875, 240.06640625], [1597.894775390625, 240.06640625],
         [1597.894775390625, 216.21875], [1811.0526123046875, 216.21875]],
        [[391.5789489746094, 310.01953125], [391.5789489746094, 279.8125], [781.5789794921875, 279.8125],
         [781.5789794921875, 310.01953125]],
        [[334.7368469238281, 310.01953125], [334.7368469238281, 286.171875], [390.0, 286.171875],
         [390.0, 310.01953125]],
        [[1779.3233642578125, 329.61846923828125], [1180.5323486328125, 316.36737060546875],
         [1181.2623291015625, 282.92694091796875], [1780.053466796875, 296.17803955078125]],
        [[429.47369384765625, 354.53515625], [429.47369384765625, 324.328125],
         [888.9473876953125, 324.328125], [888.9473876953125, 354.53515625]],
        [[183.15789794921875, 354.53515625], [183.15789794921875, 330.6875], [371.0526428222656, 330.6875],
         [371.0526428222656, 354.53515625]],
        [[1779.4737548828125, 360.89453125], [1181.0526123046875, 360.89453125],
         [1181.0526123046875, 330.6875], [1779.4737548828125, 330.6875]],
        [[1779.4737548828125, 392.69140625], [1181.0526123046875, 392.69140625],
         [1181.0526123046875, 362.484375], [1779.4737548828125, 362.484375]],
        [[183.1579132080078, 405.41015625], [183.1579132080078, 375.203125],
         [377.3684387207031, 375.203125], [377.3684387207031, 405.41015625]],
        [[895.26318359375, 399.05078125], [404.2105407714844, 399.05078125],
         [404.2105407714844, 375.203125], [895.26318359375, 375.203125]],
        [[1053.157958984375, 405.41015625], [896.8421020507812, 405.41015625],
         [896.8421020507812, 381.5625], [1053.157958984375, 381.5625]],
        [[1356.3157958984375, 424.48828125], [1181.0526123046875, 424.48828125],
         [1181.0526123046875, 400.640625], [1356.3157958984375, 400.640625]],
        [[1357.894775390625, 430.84765625], [1357.894775390625, 400.640625],
         [1779.4737548828125, 400.640625], [1779.4737548828125, 430.84765625]],
        [[404.2105407714844, 449.92578125], [404.2105407714844, 419.71875], [977.368408203125, 419.71875],
         [977.368408203125, 449.92578125]],
        [[183.1579132080078, 449.9257507324219], [183.1579132080078, 426.0780944824219],
         [383.6842041015625, 426.0780944824219], [383.6842041015625, 449.9257507324219]],
        [[157.89474487304688, 500.8007507324219], [157.89474487304688, 470.5937194824219],
         [560.5262451171875, 470.5937194824219], [560.5262451171875, 500.8007507324219]],
        [[606.3157958984375, 500.80078125], [606.3157958984375, 470.59375], [737.368408203125, 470.59375],
         [737.368408203125, 500.80078125]],
        [[814.7368774414062, 500.80078125], [814.7368774414062, 470.59375], [870.0, 470.59375],
         [870.0, 500.80078125]], [[1608.9473876953125, 500.80078125], [1547.368408203125, 500.80078125],
                                  [1547.368408203125, 476.953125], [1608.9473876953125, 476.953125]],
        [[1823.6842041015625, 500.80078125], [1698.9473876953125, 500.80078125],
         [1698.9473876953125, 476.953125], [1823.6842041015625, 476.953125]],
        [[1015.26318359375, 507.16015625], [947.368408203125, 507.16015625], [947.368408203125, 483.3125],
         [1015.26318359375, 483.3125]],
        [[56.84209442138672, 596.1913452148438], [56.84209442138672, 508.7499694824219],
         [86.84209442138672, 508.7499694824219], [86.84209442138672, 596.1913452148438]],
        [[132.63156127929688, 538.9569702148438], [132.63156127929688, 508.7499694824219],
         [427.8946533203125, 508.7499694824219], [427.8946533203125, 538.9569702148438]],
        [[503.6842041015625, 538.95703125], [442.1052551269531, 538.95703125],
         [442.1052551269531, 515.109375], [503.6842041015625, 515.109375]],
        [[642.631591796875, 538.95703125], [517.8947143554688, 538.95703125],
         [517.8947143554688, 515.109375], [642.631591796875, 515.109375]],
        [[718.4210815429688, 538.95703125], [644.2105102539062, 538.95703125],
         [644.2105102539062, 515.109375], [718.4210815429688, 515.109375]],
        [[1128.9473876953125, 538.95703125], [1092.631591796875, 538.95703125],
         [1092.631591796875, 521.46875], [1128.9473876953125, 521.46875]],
        [[1143.1578369140625, 538.9569702148438], [1143.1578369140625, 515.1093139648438],
         [1242.6314697265625, 515.1093139648438], [1242.6314697265625, 538.9569702148438]],
        [[1427.368408203125, 538.9569702148438], [1427.368408203125, 515.1093139648438],
         [1514.2105712890625, 515.1093139648438], [1514.2105712890625, 538.9569702148438]],
        [[1560.0, 538.95703125], [1560.0, 515.109375], [1602.631591796875, 515.109375],
         [1602.631591796875, 538.95703125]],
        [[1855.26318359375, 545.31640625], [1787.368408203125, 545.31640625],
         [1787.368408203125, 521.46875], [1855.26318359375, 521.46875]],
        [[345.78948974609375, 570.75390625], [132.63157653808594, 570.75390625],
         [132.63157653808594, 546.90625], [345.78948974609375, 546.90625]],
        [[353.6842041015625, 570.7538452148438], [353.6842041015625, 546.9061889648438],
         [535.26318359375, 546.9061889648438], [535.26318359375, 570.7538452148438]],
        [[472.1052551269531, 608.91015625], [132.63157653808594, 608.91015625],
         [132.63157653808594, 585.0625], [472.1052551269531, 585.0625]],
        [[86.84210968017578, 666.14453125], [63.157894134521484, 666.14453125], [63.157894134521484, 610.5],
         [86.84210968017578, 610.5]], [[132.63157653808594, 647.06640625], [132.63157653808594, 616.859375],
                                       [320.52630615234375, 616.859375],
                                       [320.52630615234375, 647.06640625]],
        [[421.5789489746094, 640.70703125], [322.1052551269531, 640.70703125],
         [322.1052551269531, 623.21875], [421.5789489746094, 623.21875]],
        [[1781.052490234375, 653.42578125], [1781.052490234375, 629.578125],
         [1861.5787353515625, 629.578125], [1861.5787353515625, 653.42578125]],
        [[339.47369384765625, 678.86328125], [132.63157653808594, 678.86328125],
         [132.63157653808594, 655.015625], [339.47369384765625, 655.015625]],
        [[408.9473876953125, 678.86328125], [353.6842041015625, 678.86328125],
         [353.6842041015625, 655.015625], [408.9473876953125, 655.015625]],
        [[535.26318359375, 678.86328125], [416.84210205078125, 678.86328125],
         [416.84210205078125, 655.015625], [535.26318359375, 655.015625]],
        [[56.842105865478516, 717.01953125], [56.842105865478516, 674.09375],
         [86.84210968017578, 674.09375], [86.84210968017578, 717.01953125]],
        [[472.1052551269531, 717.01953125], [132.63157653808594, 717.01953125],
         [132.63157653808594, 686.8125], [472.1052551269531, 686.8125]],
        [[56.84209442138672, 977.75390625], [56.84209442138672, 718.609375],
         [86.84209442138672, 718.609375], [86.84209442138672, 977.75390625]],
        [[1520.5263671875, 837.84765625], [1370.5263671875, 837.84765625], [1370.5263671875, 814.0],
         [1520.5263671875, 814.0]],
        [[1867.894775390625, 844.20703125], [1743.157958984375, 844.20703125], [1743.157958984375, 814.0],
         [1867.894775390625, 814.0]],
        [[446.84210205078125, 901.44140625], [202.1052703857422, 901.44140625],
         [202.1052703857422, 871.234375], [446.84210205078125, 871.234375]],
        [[939.4736938476562, 895.08203125], [650.5263061523438, 895.08203125],
         [650.5263061523438, 871.234375], [939.4736938476562, 871.234375]],
        [[1501.5789794921875, 907.80078125], [1402.105224609375, 907.80078125],
         [1402.105224609375, 877.59375], [1501.5789794921875, 877.59375]],
        [[1697.368408203125, 901.44140625], [1534.73681640625, 901.44140625], [1534.73681640625, 877.59375],
         [1697.368408203125, 877.59375]],
        [[372.6316223144531, 958.6757202148438], [372.6316223144531, 934.8280639648438],
         [756.3157958984375, 934.8280639648438], [756.3157958984375, 958.6757202148438]],
        [[290.52630615234375, 965.0350952148438], [290.52630615234375, 941.1874389648438],
         [371.0526123046875, 941.1874389648438], [371.0526123046875, 965.0350952148438]],
        [[1181.0526123046875, 965.03515625], [1181.0526123046875, 941.1875], [1381.5789794921875, 941.1875],
         [1381.5789794921875, 965.03515625]],
        [[1476.3157958984375, 965.03515625], [1376.8421630859375, 965.03515625],
         [1376.8421630859375, 941.1875], [1476.3157958984375, 941.1875]],
        [[1484.2105712890625, 965.03515625], [1484.2105712890625, 941.1875], [1646.8421630859375, 941.1875],
         [1646.8421630859375, 965.03515625]],
        [[1649.089111328125, 1082.8670654296875], [1630.26318359375, 988.087890625],
         [1887.1458740234375, 936.3568725585938], [1905.9718017578125, 1031.135986328125]],
        [[371.0526428222656, 1003.19140625], [176.84210205078125, 1003.19140625],
         [176.84210205078125, 979.34375], [371.0526428222656, 979.34375]],
        [[435.78948974609375, 1009.55078125], [435.78948974609375, 979.34375],
         [901.5789794921875, 979.34375], [901.5789794921875, 1009.55078125]],
        [[370.8219909667969, 1048.0826416015625], [176.0263671875, 1041.3192138671875],
         [177.042724609375, 1011.640869140625], [371.8383483886719, 1018.404296875]],
        [[397.8947448730469, 1041.34765625], [397.8947448730469, 1017.5], [465.78948974609375, 1017.5],
         [465.78948974609375, 1041.34765625]],
        [[920.5263061523438, 1047.70703125], [467.3684387207031, 1047.70703125],
         [467.3684387207031, 1023.859375], [920.5263061523438, 1023.859375]],
        [[934.7368774414062, 1041.34765625], [934.7368774414062, 1023.859375],
         [1002.631591796875, 1023.859375], [1002.631591796875, 1041.34765625]],
        [[1078.3822021484375, 1048.136962890625], [1002.6445922851562, 1041.2042236328125],
         [1004.56005859375, 1019.989013671875], [1080.2977294921875, 1026.921630859375]],
        [[176.84210205078125, 1085.86328125], [176.84210205078125, 1055.65625],
         [371.0526428222656, 1055.65625], [371.0526428222656, 1085.86328125]],
        [[1831.5789794921875, 1085.86328125], [1831.5789794921875, 1055.65625],
         [1880.5263671875, 1055.65625], [1880.5263671875, 1085.86328125]],
        [[730.885986328125, 1097.7579345703125], [397.3167419433594, 1087.5799560546875],
         [398.1946105957031, 1058.411376953125], [731.7637939453125, 1068.58935546875]],
        [[776.8421020507812, 1092.22265625], [776.8421020507812, 1068.375], [1084.73681640625, 1068.375],
         [1084.73681640625, 1092.22265625]],
        [[126.31578063964844, 1136.7381591796875], [126.31578063964844, 1106.5311279296875],
         [320.5262756347656, 1106.5311279296875], [320.5262756347656, 1136.7381591796875]],
        [[1155.7894287109375, 1143.09765625], [1155.7894287109375, 1112.890625],
         [1255.26318359375, 1112.890625], [1255.26318359375, 1143.09765625]],
        [[631.9556884765625, 1154.7674560546875], [629.447021484375, 1119.403564453125],
         [787.5501098632812, 1108.032470703125], [790.0587768554688, 1143.3963623046875]],
        [[1135.1109619140625, 1151.603759765625], [1015.1591186523438, 1142.9766845703125],
         [1017.1547241210938, 1114.84375], [1137.1065673828125, 1123.4708251953125]],
        [[1710.1707763671875, 1179.10546875], [1703.1295166015625, 1136.56640625],
         [1859.0614013671875, 1110.3984375], [1866.1026611328125, 1152.9375]],
        [[1571.0526123046875, 1149.45703125], [1446.3157958984375, 1149.45703125],
         [1446.3157958984375, 1125.609375], [1571.0526123046875, 1125.609375]]]
    ret_dic = {'buyer_name': (329, 274, 791, 316), 'buyer_id': (424, 319, 898, 360),
               'buyer_address_phone': (399, 370, 1063, 411), 'buyer_bank_no': (399, 414, 987, 455),
               'seller_name': (285, 929, 766, 964), 'seller_id': (430, 974, 911, 1015),
               'seller_address_phone': (392, 1014, 1090, 1054), 'seller_bank_no': (392, 1053, 741, 1103),
               'bill_no': (1433, 96, 1702, 162), 'price': (1365, 809, 1530, 843), 'tax': (1738, 809, 1877, 850),
               'account_cap': (645, 866, 949, 901), 'account_lower': (1529, 872, 1707, 907),
               'date': (1592, 211, 1821, 246)}
    for i in range(3):
        s1 = time.time()
        get_perspective_img('')
        print(time.time() - s1)
