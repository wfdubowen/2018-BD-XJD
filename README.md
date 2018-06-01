# 2018-BD-XJD
<br>
<h3>2018百度西交大大数据竞赛-商家招牌的分类与检测</h3>
<br><br>
<b>【CSDN博客】</b>https://blog.csdn.net/u013063099/article/details/80533694
<br><br>
<b>【运行环境】</b>Windows 10，PyTorch 0.4.0
<br><br>
<b>【简介】使用单模型ResNet152</b>
<br>
第1步 数据进行预处理（utils/image_preprocessing.py），将处理之后的数据存在dataset/data中。<br>
第2步 进行训练，训练过程中的模型权重以pth文件形式保存在output文件夹中。<br>
第3步 预测，使用保存的pth模型文件，最后生成的csv保存在output文件夹下。<br>
<br>
<b>线上评分0.97+</b>

