# coding=utf-8
import pandas as pd
from dataset.dataset import collate_fn, dataset
import os
import datetime
import numpy as np
from torch.autograd import Variable
import torch
import torch.utils.data as torchdata
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
import logging
import models.models as md

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
save_dir = './output/Resnet152-All-Trained-SGD-V4-Gray'
# rawdata_root = './dataset/data/train_improve_v4'
rawdata_root = './dataset/data/grayImage_v4'  # OpenCV Resized
all_pd = pd.read_csv("./dataset/data/train_improve_v4.txt", sep=" ", header=None, names=['ImageName', 'label'])
# all trained
train_pd = all_pd
# train_pd, val_pd = train_test_split(all_pd, test_size=0.15, random_state=43, stratify=all_pd['label'])

# Normal's transforms
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomRotation(degrees=15),
        # transforms.Resize(224),
        # transforms.RandomResizedCrop(224, scale=(0.49, 1.0)),
        transforms.ToTensor(),  # 0-255 to 0-1
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# PNASnet's transforms
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.RandomRotation(degrees=15),
#         transforms.RandomResizedCrop(299, scale=(0.49, 1.0)),
#         transforms.ToTensor(),  # 0-255 to 0-1
#         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
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
          print_inter=200):

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
            # Multi GPUs
            # net = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
            # outputs = net(inputs)

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
                batch_acc = 1.0 * batch_corrects / (labels.size(0))

                logging.info('%s [%d-%d] | batch-loss: %.5f | acc@1: %.5f'
                             % (dt(), epoch, batch_cnt, loss.item(), batch_acc))
                # save model
                save_path = os.path.join(save_dir, 'weights-%d-%d-[%.5f].pth' % (epoch, batch_cnt, loss.item()))
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
    dataloader = {}
    dataloader['train'] = torch.utils.data.DataLoader(data_set['train'], batch_size=4,
                                               shuffle=True, num_workers=4, collate_fn=collate_fn)
    '''model'''
    # model = md.Modified_Densenet169(num_classs=100)
    model = md.Modified_Resnet152(num_classs=100)
    # model = md.Modified_SENet154(num_classs=100)

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
          exp_lr_scheduler=exp_lr_scheduler, data_set=data_set, data_loader=dataloader, save_dir=save_dir)
