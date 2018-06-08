#-*-coding:utf-8-*-
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# https://github.com/Cadene/pretrained-models.pytorch
import pretrainedmodels

class Modified_Densenet169(nn.Module):
    """docstring for ClassName"""
    def __init__(self, num_classs=100):
        super(Modified_Densenet169, self).__init__()
        model = torchvision.models.densenet169(pretrained=True)
        self.num_classs = num_classs
        self.avgpool_size = 7    # for 224x224 input
        for i, m in enumerate(model.children()):
            if i == 0:
                self.features = m
            else:
                self.classifier = nn.Linear(in_features=1664, out_features=num_classs)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out

class Modified_Densenet201(nn.Module):
    """docstring for ClassName"""
    def __init__(self, num_classs=100):
        super(Modified_Densenet201, self).__init__()
        model = torchvision.models.densenet201(pretrained=True)
        self.num_classs = num_classs
        self.avgpool_size = 7    # for 224x224 input
        for i, m in enumerate(model.children()):
            if i == 0:
                self.features = m
            else:
                self.classifier = nn.Linear(in_features=1920, out_features=num_classs)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        out = self.classifier(out)
        return out

class Modified_Resnet50(nn.Module):
    """docstring for ClassName"""
    def __init__(self, num_classs=100):
        super(Modified_Resnet50, self).__init__()
        model = torchvision.models.resnet50(pretrained=True)
        self.num_classs = num_classs
        temp = []
        for i, m in enumerate(model.children()):
            if i <= 8:
                temp.append(m)
            else:
                self.classifier = nn.Linear(in_features=2048, out_features=num_classs)
        self.features = nn.Sequential(*temp)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Modified_Resnet101(nn.Module):
    """docstring for ClassName"""
    def __init__(self, num_classs=100):
        super(Modified_Resnet101, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.num_classs = num_classs
        temp = []
        for i, m in enumerate(model.children()):
            if i <= 8:
                temp.append(m)
            else:
                self.classifier = nn.Linear(in_features=2048, out_features=num_classs)
        self.features = nn.Sequential(*temp)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Modified_Resnet152(nn.Module):
    """docstring for ClassName"""
    def __init__(self, num_classs=100):
        super(Modified_Resnet152, self).__init__()
        model = torchvision.models.resnet152(pretrained=True)
        self.num_classs = num_classs
        temp = []
        for i, m in enumerate(model.children()):
            if i <= 8:
                temp.append(m)
            else:
                self.classifier = nn.Linear(in_features=2048, out_features=num_classs)
        self.features = nn.Sequential(*temp)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Modified_InceptionResnetV2(nn.Module):
    """docstring for ClassName"""
    def __init__(self, num_classs=100):
        super(Modified_InceptionResnetV2, self).__init__()
        model = pretrainedmodels.inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        self.num_classs = num_classs
        temp = []
        for i, m in enumerate(model.children()):
            if i <= 15:
                temp.append(m)
            else:
                self.classifier = nn.Linear(in_features=1536, out_features=num_classs)
        self.features = nn.Sequential(*temp)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Modified_SENet154(nn.Module):
    """docstring for ClassName"""
    def __init__(self, num_classs=100):
        super(Modified_SENet154, self).__init__()
        model = pretrainedmodels.senet154(num_classes=1000, pretrained='imagenet')
        self.num_classs = num_classs
        temp = []
        for i, m in enumerate(model.children()):
            if i <= 6:
                temp.append(m)
            else:
                self.classifier = nn.Linear(in_features=2048, out_features=num_classs)
        self.features = nn.Sequential(*temp)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class Modified_PNASnet(nn.Module):
    """docstring for ClassName"""
    def __init__(self, num_classs=100):
        super(Modified_PNASnet, self).__init__()
        model = pretrainedmodels.pnasnet5large(num_classes=1000, pretrained='imagenet')
        self.num_classs = num_classs
        temp = []
        for i, m in enumerate(model.children()):
            if i <= 17:
                temp.append(m)
            else:
                self.classifier = nn.Linear(in_features=4320, out_features=num_classs)
        self.features = nn.Sequential(*temp)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# for pytorch v0.4 python 3.5
# class Modified_nasnetalarge(object):
#     """docstring for ClassName"""
#     def __init__(self, num_classs=100):
#         super(Modified_nasnetalarge, self).__init__()
#         self.num_classs = num_classs
#         model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
#         temp = []
#         for i, m in enumerate(model.children()):
#             if i <= 25:
#                 temp.append(m)
#             else:
#                 self.classifier = nn.Linear(in_features=4032, out_features=num_classs)
#         self.features = nn.Sequential(*temp)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), -1)
#         x = self.classifier(x)
#         return x

