import pandas as pd
import numpy as np
from scipy import stats

list = []
one_pd = pd.read_csv("./../output/Resnet152-All-Trained-SGD-V4-Gray_test.csv", sep=" ", header=None, names=['ImageName', 'label'])
two_pd = pd.read_csv("./../output/SENet154-All-Trained-SGD-V4_test.csv", sep=" ", header=None, names=['ImageName', 'label'])
three_pd = pd.read_csv("./../output/Resnet152-Other-All-TrainedV2-SGD-V4-out_test.csv", sep=" ", header=None, names=['ImageName', 'label'])
new_pd = pd.read_csv("./../dataset/data/test.txt", sep=" ", header=None, names=['ImageName'])

for i in range(1000):
    one = int(one_pd.label[i])
    two = int(two_pd.label[i])
    three = int(three_pd.label[i])
    nums = [one, two, three]
    counts = np.bincount(nums)
    meta = np.argmax(counts)
    if(one!=two!=three):
        list.append(one)
    else:
        list.append(meta)

new_pd['label'] = list
print(new_pd)
new_pd[['ImageName', 'label']].to_csv('./../output/three-models-voting.csv', sep=" ", header=None, index=False)
