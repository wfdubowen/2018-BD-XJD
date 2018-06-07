# 2018-BD-XJD
<br>
<h3>2018百度西交大大数据竞赛-商家招牌的分类与检测</h3>
<br><br>
<b>【CSDN博客】</b>https://blog.csdn.net/u013063099/article/details/80533694
<br><br>
<b>【运行环境】</b>Windows 10，PyTorch 0.4.0
<br><br>
<b>【简介】单模型ResNet152最高线上评分0.99</b>
<br>
<b>第一步</b> 数据进行预处理（utils/image_preprocessing.py），根据CSDN博客中的处理阶段（V1.1-V1.4）将处理之后的数据存在dataset/data中（例如第四阶段V1.4我存到文件夹train_improve_v4中）。根据需要可以对扩增的数据进行调整图片大小、灰度化等等。然后将原来的train.txt其中的原图片名可以加上不同的前缀代表预处理之后的新图片名，label不用改。用Excel等工具进行扩充（各种复制粘贴……），形成新的label文件（例如train_improve_v4.txt）。<br>
<b>第二步</b> 进行训练，使用新扩增的训练集以及label文件进行训练。训练过程中的模型权重以pth文件形式保存在output文件夹中。后来可能发现带着验证集的效果并不好，干脆直接去掉验证集，把所有的训练集图片全部训练。全部训练使用train_only.py，带验证集的训练使用train_val.py。<br>
<b>第三步</b> 预测，使用保存的pth模型文件（保存时文件名会带有loss值标识），选择一个最低的损失值的模型（通常选损失为0.00001的），最后生成的csv保存在output文件夹下。提交等待线上评分即可。<br>
<br>
<b>用了个比较脑残的投票法，投了个0.991。。。=.=|||</b>
<br><br>
<b>目前线上最高评分0.991。</b>

