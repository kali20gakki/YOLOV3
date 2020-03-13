#!/usr/bin/python3
# -*- encoding: utf-8 -*-
'''
@File    :   gen_list_txt.py
@Time    :   2020/03/13 15:43:47
@Author  :   Mrtutu 
@Version :   1.0
@Contact :   zhangwei3.0@qq.com
@License :   
@Desc    :   生成训练和测试集文件txt
'''

# here put the import lib
import sys,os 



def generate_list_txt(xml_path, img_path, save_path):

    # 遍历图片文件夹
    for img_root, _, img_files in os.walk(img_path):
        pass
    
    img_list = []
    for img in img_files:
        # 图片序号
        img_list.append(img.split('.')[0])
    
    # 图片格式
    img_type = '.' + img_files[0].split('.')[1]
        
    # 遍历标注文件文件夹
    for xml_root, _, xml_files in os.walk(xml_path):
        pass
    
    cnt = 0
    with open(save_path, 'w') as f:
        for xml_file in xml_files:
            temp = xml_file.split('.')[0]
            if temp in img_list:
                # 图片路径 xml路径
                line = os.path.join(img_root, temp + img_type) + ' ' + os.path.join(xml_root, xml_file)
                f.write(line+'\n')
                print(line)
                cnt += 1
            else:
                print('not find %s.jpeg'%temp)
    
    print('wirte %d lines sucessfully!'%cnt)
    


if __name__ == '__main__':
    train_xml_path = 'insects/train/annotations/xmls'
    train_img_path = 'insects/train/images'
    
    val_xml_path = 'insects/val/annotations/xmls'
    val_img_path = 'insects/val/images'
    
    train_list_save_path = 'train_list.txt'
    val_list_save_path = 'val_list.txt'
    
    
    generate_list_txt(train_xml_path, train_img_path, train_list_save_path)
    generate_list_txt(val_xml_path, val_img_path, val_list_save_path)