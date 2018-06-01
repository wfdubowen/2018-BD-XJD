from PIL import Image, ImageEnhance, ImageFilter
import os
import shutil


CONTRAST_FLAGE = True                     # 对比度标志位
LIGHT_FLAGE = True                        # 亮度增强标志位
COLOR_FLAGE = True                        # 色彩增强标志位
MIRROR_FLAGE = True                       # 镜像标志位
TRANSPOSE_FLAGE = True                    # 旋转图像标志位
CUT_FLAGE = True                          # 裁剪标志位
GAUSSIAN_BLUR = True                      # 高斯模糊标志位

work_path = 'image_conver'                # 存放要工作文件夹名
work_dir_name = '1'                       # 存放图像文件的文件夹名
convert_image_dir_name = 'convert_image'  # 存放转化后图片的文件夹名

cwd = os.getcwd()                                                                    # 取得当前路径
work_dir_path = cwd + os.sep + work_path + os.sep + work_dir_name                    # 得到存放图像文件的文件夹路径
convert_image_dir_path = cwd + os.sep + work_path + os.sep + convert_image_dir_name  # 得到存放转化后的图像文件的文件夹路径

'''高斯模糊'''
class MyGaussianBlur(ImageFilter.Filter):

    name = "GaussianBlur"

    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:
            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)


image_number = 0                                      # 转换图片的数量
if os.path.exists(convert_image_dir_path):            # 如果convert_image文件夹存在，则递归地删除convert_image文件夹
    shutil.rmtree(convert_image_dir_path)
os.mkdir(convert_image_dir_path)                      # 生成convert_image文件夹
image_file_name = os.listdir(work_dir_path)           # 得到每个图像文件的文件名list
image_file_name_path = []                             # 保存要转化图像的绝对路径的list

for x in image_file_name:
    image_file_name_path.append(work_dir_path + os.sep + x)                        # 得到每个图像文件的绝对路径list

for x in image_file_name_path:
    img = Image.open(x)                                                            # 打开图像，得到Image对象

    if MIRROR_FLAGE:                                                               # 对图像进行镜像
        img_mirror = img.transpose(Image.FLIP_LEFT_RIGHT)
        img_mirror.save(convert_image_dir_path + os.sep + 'mirror_' + os.path.basename(x))
    if CUT_FLAGE:                                                                  # 对图像进行裁剪
        img_cut = img.crop((100, 100, 350, 350))
        img_cut.save(convert_image_dir_path + os.sep + 'cut_' + os.path.basename(x))
    if GAUSSIAN_BLUR:
        img_gb = img.filter(MyGaussianBlur(radius=4))                              # 对图像进行高斯模糊，模糊半径为4
        img_gb.save(convert_image_dir_path + os.sep + 'gb_4_' + os.path.basename(x))
    if TRANSPOSE_FLAGE:                                                            # 对图像进行旋转
        img_rotate_30 = img.rotate(30)                                             # 对图像旋转30度
        img_rotate_negative_30 = img.rotate(-30)                                   # 对图像旋转负30度
        img_rotate_30.save(convert_image_dir_path + os.sep + 'rotate_30_' + os.path.basename(x))
        img_rotate_negative_30.save(convert_image_dir_path + os.sep + 'rotate_negative_30_' + os.path.basename(x))
    if LIGHT_FLAGE:                                                                # 对图像进行亮度增强
        enh_bri = ImageEnhance.Brightness(img)
        brightness = 2
        img_brightened = enh_bri.enhance(brightness)
        img_brightened.save(convert_image_dir_path + os.sep + 'light_2_' + os.path.basename(x))
    if COLOR_FLAGE:                                                                # 对图像进行色彩增强
        enh_col = ImageEnhance.Color(img)
        color = 2
        img_colored = enh_col.enhance(color)
        img_colored.save(convert_image_dir_path + os.sep + 'color_2_' + os.path.basename(x))
    if CONTRAST_FLAGE:                                                             # 对图像进行对比度增强
        enh_con = ImageEnhance.Contrast(img)
        contrast = 2
        img_contrast = enh_con.enhance(contrast)
        img_contrast.save(convert_image_dir_path + os.sep + 'contrast_2_' + os.path.basename(x))

    image_number += 1
    # 显示处理到第几张,尺寸，图像模式
    print("convert pictur" "es :%s size:%s mode:%s" % (image_number, img.size, img.mode))
