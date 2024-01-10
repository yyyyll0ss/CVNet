import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import os.path as osp
import os
from tqdm import tqdm
from CVNet.utils.visualizer import save_viz
# from Hyper import get_dataset_rgb, color_map
from tools.test_pipelines import generate_coco_ann_DP, generate_coco_ann
import json
import time


def save_per_class_result(per_class_result,output_dir,img_name):
    """
    save different result to .png
    """
    file_path = output_dir
    if not osp.exists(file_path):
        os.makedirs(file_path)
    cv2.imwrite(osp.join(file_path,img_name),per_class_result)

def draw_polygon(all_class_polygon,save_path,image_name):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patches as Patches
    import os.path as osp
    import os
    from skimage import io

    colors = color_map(dataset).tolist()
    image = io.imread(f'/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/{dataset}_rgb.png')
    plt.axis('off')
    plt.imshow(image)

    for key,values in all_class_polygon.items():
        for polygon in values:
            polygon = np.array(polygon)
            color = np.array(colors[int(key)-1])/255
            plt.gca().add_patch(Patches.Polygon(polygon, fill=False, ec=color, linewidth=0.5))
            plt.plot(polygon[:,0], polygon[:,1], color=color, marker='.', linewidth=0.75,markersize=2)   # linewidth=2,markersize=3.5

    file_path = osp.join(save_path, dataset)
    if not osp.exists(file_path):
        os.makedirs(file_path)
    impath = osp.join(file_path, f'{dataset}_{numTrain}_polygon_result.pdf')

    plt.savefig(impath, bbox_inches='tight', pad_inches=0.0, dpi=800)
    plt.clf()


import math

class Point(object):
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y
    def __str__(self):
        return '{} {} {}'.format(self.id,self.x,self.y)

class DPCompress(object):
    def __init__(self, pointList, tolerance):
        self.Compressed = list()
        self.pointList = pointList
        self.tolerance = tolerance
        self.runDP(pointList, tolerance)

    def calc_height(self, point1, point2, point):
        """
        计算point到[point1, point2所在直线]的距离
        点到直线距离：
        A = point2.y - point1.y;
        B = point1.x - point2.x;
        C = point2.x * point1.y - point1.x * point2.y
        Dist = abs((A * point3.X + B * point3.Y + C) / sqrt(A * A + B * B))
        """
        tops = abs(point1.x * point.y + point2.x * point1.y + point.x * point2.y
                - point1.x * point2.y - point2.x * point.y - point.x * point1.y
                )

        bottom = math.sqrt(
            math.pow(point2.y - point1.y, 2) + math.pow(point2.x - point1.x, 2)
        )

        height = tops / bottom
        return height

    def DouglasPeucker(self, pointList, firsPoint, lastPoint, tolerance):
        """
        计算通过的内容
        DP算法
        :param pointList: 点列表
        :param firsPoint: 第一个点
        :param lastPoint: 最后一个点
        :param tolerance: 容差
        :return:
        """
        maxDistance = 0.0
        indexFarthest = firsPoint
        for i in range(firsPoint, lastPoint):
            distance = self.calc_height(pointList[firsPoint], pointList[lastPoint], pointList[i])
            if (distance > maxDistance):
                maxDistance = distance
                indexFarthest = i
    #    print('max_dis=', maxDistance)

        if maxDistance > tolerance and indexFarthest != firsPoint:
            self.Compressed.append(pointList[indexFarthest])
            self.DouglasPeucker(pointList, firsPoint, indexFarthest, tolerance)
            self.DouglasPeucker(pointList, indexFarthest, lastPoint, tolerance)

    def runDP(self, pointList, tolerance):
        """
        主要运行结果
        :param pointList: Point 列表
        :param tolerance: 值越小，压缩后剩余的越多
        :return:
        """
        if pointList == None or pointList.__len__() < 3:
            return pointList

        firspoint = 0
        lastPoint = len(pointList) - 1

        self.Compressed.append(pointList[firspoint])
        self.Compressed.append(pointList[lastPoint])

        self.DouglasPeucker(pointList, firspoint, lastPoint, tolerance)

    def getCompressed(self):
        self.Compressed.sort(key=lambda point: int(point.id))
        return self.Compressed






if __name__ == "__main__":
    # load image data
    root = '/home/isalab301/yyl/VecCD/snunet_c32_infer_levir'
    img_root = '/home/isalab301/yyl/VecCD/changeformer_b4_infer_levir/vis_image'
    img_bg_root = '/home/isalab301/yyl/VecCD/data/LEVIR_VectorCD/val'
    save_path = '/home/isalab301/yyl/VecCD/run_test/output'
    # all_result_output_dir = '/home/isalab301/yyl/HSI-poly-new/HSI-poly-new/polygon_result'

    sum_contour = 0
    sum_poly = 0
    results = []
    # print(os.listdir(img_root))

    time_start = time.time()
    for i in tqdm(os.listdir(img_root)):
        img = io.imread(osp.join(img_root,i))
        if len(img.shape) == 3:
            img = img[:,:,0]   #(300,300,1)

        #save counter
        _, thresh = cv2.threshold(img * 255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        per_class_mask_bgr = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        Counter_res = cv2.drawContours(per_class_mask_bgr,contours,-1,(0,255,0),1)
        #save_per_class_result(Counter_res,countour_output_dir,i)   #####

        polygons = []

        for contour in contours:
            if cv2.contourArea(contour) < 20:   # indian,salinas(40),HongHu,LongKou(100)
                continue
            if len(contour) < 3:
                continue
            # 2.进行多边形逼近，得到多边形的角点
            epsilon = 0.06 * cv2.arcLength(contour,True)   #0.03 smaller->complex   0.035
            # print('countour points:',len(contour))
            sum_contour += len(contour)
            # approx = cv2.approxPolyDP(contour, epsilon, True)
            point_list = []
            for j,coord in enumerate(contour):
                point_list.append(Point(j,coord[0][0],coord[0][1]))
            dp = DPCompress(point_list,epsilon)
            approx = []
            for p in dp.getCompressed():
                approx.append([p.x,p.y])
            approx = np.array(approx)
            # print('approx points:', len(approx))
            sum_poly += len(approx)
            # 3.画出多边形
            # cv2.polylines(per_class_mask_bgr, [approx], True, (0, 255, 0), 1)
            if len(approx) > 2:
                approx = approx.squeeze().tolist()
                approx.append(approx[0])
                polygons.append(np.array(approx))

        # img_bg = io.imread(osp.join(img_bg_root,'B', i))
        # save_viz(image=img_bg, polys=polygons, save_path=save_path, filename=i)

        #get DP coco ann results
        image_result = generate_coco_ann_DP(polygons, i)
        if len(image_result) != 0:
            results.extend(image_result)
    # creat polygon save path
    poly_path_ = osp.join(root,'DP_output')   #####
    if not osp.exists(poly_path_):
        os.makedirs(poly_path_)
    poly_path = osp.join(poly_path_, f'VectorCD_DP.json')   #####

    # save two style result to json file
    with open(poly_path, 'w') as _out:
        json.dump(results, _out)

    time_end = time.time()
    time_mean = (time_end - time_start)*1000/len(os.listdir(img_root))
    print(time_mean)













