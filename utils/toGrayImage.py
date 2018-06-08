import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--Read_context", help = '需要转换的目录路径，以/结束', default = './../dataset/data/traslateImg_v4/')
parser.add_argument("--Write_context", help = '保存目录路径，以/结束', default = './../dataset/data/train_improve_v5/')
args = parser.parse_args()



num = 0 #记录数量

if os.path.exists(args.Read_context) == False:
    print(args.Read_context, "文件夹不存在，退出")
    os._exit()

if os.path.exists(args.Write_context) == False:
    print(args.Write_context,"文件夹不存在，创建文件夹")
    os.makedirs(args.Write_context)

for filename in os.listdir(args.Read_context):
    image = cv2.imread(args.Read_context + str(filename), 0)
    cv2.imwrite(args.Write_context + str("gray_"+filename), image)
    num = num + 1
    print(filename, '已经完成, num = ', num)

cv2.destroyAllWindows()

