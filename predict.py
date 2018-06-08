import os
import numpy as np
import pandas as pd
from dataset.dataset import dataset, collate_fn
import torch
from torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from math import ceil
from torch.nn.functional import softmax
import models.models as md

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

rawdata_root = './dataset/data/grayImage_test'
# rawdata_root = './dataset/data/translate_test'
true_test_pb = pd.read_csv("./dataset/data/test.txt", sep=" ", header=None, names=['ImageName'])
true_test_pb['label'] = 1

output_name = 'Resnet152-All-Trained-SGD-V4-Gray'
resume = './output/Resnet152-All-Trained-SGD-V4-Gray/weights-13-234-[0.00001].pth'

# Normal's transforms
test_transforms = transforms.Compose([
                # transforms.Resize(224),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

# PNASnet's transforms
# test_transforms = transforms.Compose([
#                 transforms.Resize(299),
#                 transforms.CenterCrop(299),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#             ])

if __name__ == '__main__':

    test_pd = true_test_pb
    # print(test_pd.head())

    data_set = {}
    data_set['test'] = dataset(imgroot=rawdata_root, anno_pd=test_pd,
                               transforms=test_transforms)
    data_loader = {}
    data_loader['test'] = torchdata.DataLoader(data_set['test'], batch_size=4, num_workers=4,
                                           shuffle=False, pin_memory=True, collate_fn=collate_fn)

    # model = md.Modified_Densenet169(num_classs=100)
    model = md.Modified_Resnet152(num_classs=100)
    # model = md.Modified_SENet154(num_classs=100)

    print('Resuming finetune from %s' % resume)
    model.load_state_dict(torch.load(resume))
    model = model.cuda()
    model.eval()
    criterion = CrossEntropyLoss()

    if not os.path.exists('./output'):
        os.makedirs('./output')

    test_size = ceil(len(data_set['test']) / data_loader['test'].batch_size)
    test_preds = np.zeros((len(data_set['test'])), dtype=np.float32)
    true_label = np.zeros((len(data_set['test'])), dtype=np.int)
    idx = 0
    test_loss = 0
    test_corrects = 0
    for batch_cnt_test, data_test in enumerate(data_loader['test']):
        # print data
        print("{0}/{1}".format(batch_cnt_test + 1, int(test_size)))
        inputs, labels = data_test
        inputs = Variable(inputs.cuda())
        labels = Variable(torch.from_numpy(np.array(labels)).long().cuda())
        # forward
        outputs = model(inputs)

        # statistics
        if isinstance(outputs, list):
            loss = criterion(outputs[0], labels)
            loss += criterion(outputs[1], labels)
            outputs = (outputs[0]+outputs[1])/2
        else:
            loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        test_loss += loss.item()
        batch_corrects = torch.sum((preds == labels)).item()
        test_corrects += batch_corrects
        test_preds[idx:(idx + labels.size(0))] = preds
        true_label[idx:(idx + labels.size(0))] = labels.data.cpu().numpy()
        # statistics
        idx += labels.size(0)
    test_loss = test_loss / test_size
    test_acc = 1.0 * test_corrects / len(data_set['test'])
    print('test-loss: %.5f ||test-acc: %.5f' % (test_loss, test_acc))

    test_pred = test_pd[['ImageName']].copy()
    test_pred['label'] = list(test_preds)
    test_pred['label'] = test_pred['label'].apply(lambda x: int(x)+1)
    test_pred[['ImageName', "label"]].to_csv('./output/{0}_test.csv'.format(output_name), sep=" ", header=None, index=False)
    print("Done")
