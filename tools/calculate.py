#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File    :   calculate.py
@Time    :   2020/03/13 15:48:04
@Author  :   Mrtutu 
@Version :   1.0
@Contact :   zhangwei3.0@qq.com
@License :   
@Desc    :   Kmeans 生成 anchor
'''

# here put the import lib
import glob
import xml.etree.ElementTree as ET

import numpy as np

from kmeans import kmeans, avg_iou

ANNOTATIONS_PATH = "../insects/train/annotations/xmls"
CLUSTERS = 9

def load_dataset(path):
	dataset = []
	for xml_file in glob.glob("{}/*xml".format(path)):
		tree = ET.parse(xml_file)

		height = int(tree.findtext("./size/height"))
		width = int(tree.findtext("./size/width"))

		for obj in tree.iter("object"):
			xmin = int(obj.findtext("bndbox/xmin")) / width
			ymin = int(obj.findtext("bndbox/ymin")) / height
			xmax = int(obj.findtext("bndbox/xmax")) / width
			ymax = int(obj.findtext("bndbox/ymax")) / height

			dataset.append([xmax - xmin, ymax - ymin])

	return np.array(dataset)

if __name__ == '__main__':
    num = input('epoch:')
    data = load_dataset(ANNOTATIONS_PATH)
    
    # 论文中的大小 ACC：63.89%
    yolov3_anchors = [[10,13],[16,30],[33,23],[30,61],[62,45],[59,119],[116,90],[156,198],[373,326]]
    yolov3_anchors = np.array(yolov3_anchors)
    print('YOLOV3 ACC:%.2f%%'%(avg_iou(data, yolov3_anchors/416)*100))
    
    a = int(data.size/2)
    print('dataset anchor num: %d'%a)
    # 用于计算anchor平均大小
    anchor_sum = np.ones((CLUSTERS, 2))*0
    # 用于计算平均acc
    acc_sum = np.ones((a,2))*0
    
    # 迭代多次
    for i in range(1,int(num)+1):
        out,t = kmeans(data, k=CLUSTERS)
        acc = avg_iou(data, out)
        out = sorted(out, key=(lambda x: x[0]))
        anchor_sum += out
        acc_sum += acc
        print('\n%d th calculate done! cost time: %.2fs Accuracy:%.2f%%'%(i, t,(acc*100)))
    
    # 平均大小
    out = anchor_sum / int(num)
    acc = (acc_sum / int(num))[0][0]
    
    print('\nRaw ouput:')
    print(out)
    print('Anchors:')
    print(np.around(sorted(out*416, key=(lambda x: x[1]*x[0]))))
    print("Average accuracy: %.2f%%"%(acc*100))
    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))
