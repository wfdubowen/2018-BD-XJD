# coding=utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset.dataset import collate_fn, dataset
import os
import time
import datetime
import numpy as np
from math import ceil
from torch.autograd import Variable
import torch
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
import logging
from models.inception_resnet_v2 import inceptionresnetv2
from models.xception import xception
from models.models import Modified_Densenet201
from models.models import Modified_Resnet152

save_dir = './output/Resnet152-SGD-V4'

rawdata_root = './dataset/data/train_improve_v4'
all_pd = pd.read_csv("./dataset/data/train_improve_v4.txt", sep=" ", header=None, names=['ImageName', 'label'])
train_pd, val_pd = train_test_split(all_pd, test_size=0.15, random_state=43, stratify=all_pd['label'])
# print(val_pd.shape)
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# Normal's transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomResizedCrop(224, scale=(0.49, 1.0)),
        transforms.ToTensor(),  # 0-255 to 0-1
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Xception's transforms
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomRotation(degrees=15),
#         transforms.RandomResizedCrop(299, scale=(0.49, 1.0)),
#         transforms.ToTensor(),  # 0-255 to 0-1
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
#     ]),
#     'val': transforms.Compose([
#         transforms.Resize(299),
#         transforms.CenterCrop(299),
#         transforms.ToTensor(),
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#     ]),
# }

def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def trainlog(logfilepath, head='%(message)s'):
    logger = logging.getLogger('mylogger')
    logging.basicConfig(filename=logfilepath, level=logging.INFO, format=head)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(head)
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def train(model,
          epoch_num,
          start_epoch,
          optimizer,
          criterion,
          exp_lr_scheduler,
          data_set,
          data_loader,
          save_dir,
          print_inter=200,
          val_inter=3500):

    step = -1
    for epoch in range(start_epoch, epoch_num):
        # train phase
        exp_lr_scheduler.step(epoch)
        model.train(True)  # Set model to training mode

        for batch_cnt, data in enumerate(data_loader['train']):

            step += 1
            model.train(True)
            # print data
            inputs, labels = data

            inputs = Variable(inputs.cuda())
            labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(inputs)
            if isinstance(outputs, list):
                loss = criterion(outputs[0], labels)
                loss += criterion(outputs[1], labels)
                outputs = outputs[0]
            else:
                loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()

            # batch loss
            if step % print_inter == 0:
                _, preds = torch.max(outputs, 1)

                batch_corrects = float(torch.sum((preds == labels)).item())
                batch_acc = batch_corrects / (labels.size(0))

                logging.info('%s [%d-%d] | batch-loss: %.5f | acc: %.2f'
                             % (dt(), epoch, batch_cnt, loss.item(), batch_acc))


            if step % val_inter == 0:
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())
                # val phase
                model.train(False)  # Set model to evaluate mode

                val_loss = 0
                val_corrects = 0
                val_size = ceil(len(data_set['val']) / data_loader['val'].batch_size)

                t0 = time.time()

                for batch_cnt_val, data_val in enumerate(data_loader['val']):
                    # print data
                    inputs,  labels = data_val

                    inputs = Variable(inputs.cuda())
                    labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())

                    # forward
                    outputs = model(inputs)
                    if isinstance(outputs, list):
                        loss = criterion(outputs[0], labels)
                        loss += criterion(outputs[1], labels)
                        outputs = outputs[0]

                    else:
                        loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # statistics
                    val_loss += loss.item()
                    batch_corrects = float(torch.sum((preds == labels)).item())
                    val_corrects += batch_corrects

                val_loss = val_loss / val_size
                val_acc = 1.0 * val_corrects / len(data_set['val'])

                t1 = time.time()
                since = t1-t0
                logging.info('--'*30)
                logging.info('current lr:%s' % exp_lr_scheduler.get_lr())
                logging.info('%s epoch[%d]-val-loss: %.5f ||val-acc: %.5f ||time: %d'
                             % (dt(), epoch, val_loss, val_acc, since))
                # save model
                save_path = os.path.join(save_dir, 'weights-%d-%d-[%.5f].pth' % (epoch, batch_cnt, val_loss))
                torch.save(model.state_dict(), save_path)
                logging.info('saved model to %s' % (save_path))
                logging.info('--' * 30)


if __name__ == '__main__':

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logfile = save_dir + '/trainlog.log'
    trainlog(logfile)

    '''data'''
    data_set = {}
    data_set['train'] = dataset(imgroot=rawdata_root, anno_pd=train_pd,
                                transforms=data_transforms["train"])
    data_set['val'] = dataset(imgroot=rawdata_root, anno_pd=val_pd,
                                transforms=data_transforms["val"])
    dataloader = {}
    dataloader['train'] = torch.utils.data.DataLoader(data_set['train'], batch_size=4,
                                               shuffle=True, num_workers=4, collate_fn=collate_fn)
    dataloader['val'] = torch.utils.data.DataLoader(data_set['val'], batch_size=4,
                                               shuffle=True, num_workers=4, collate_fn=collate_fn)
    '''model'''
    # model = inceptionresnetv2(num_classes=100, pretrained='imagenet')
    # model = xception(pretrained='imagenet')
    # model = Modified_Densenet201(num_classs=100)
    model = Modified_Resnet152(num_classs=100)

    resume = None
    if resume:
        logging.info('Resuming finetune from %s' % resume)
        model.load_state_dict(torch.load(resume))
    model = model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    # optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = CrossEntropyLoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)

    '''training'''
    train(model, epoch_num=50, start_epoch=0, optimizer=optimizer, criterion=criterion,
          exp_lr_scheduler=exp_lr_scheduler, data_set=data_set, data_loader=dataloader, save_dir=save_dir,
          print_inter=50, val_inter=400)
