import json
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os
import pickle

TRAIN_FILE = "train.p"
TEST_FILE = "test.p"

class TUMOR_IMG(Dataset):

    def __init__(self, root, train=True, transform = None, target_transform=None):
        super(TUMOR_IMG, self).__init__()
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if self.train:
            image_file = os.path.join(root, TRAIN_FILE)
            img_folder = os.path.join(root, 'train_tumor')
        else:
            image_file = os.path.join(root, TEST_FILE)
            img_folder = os.path.join(root, 'test_tumor')

        with open(image_file, mode='rb') as f:
            data = pickle.load(f)

        self.images, self.seg_images, self.labels = \
            data['images'], data["corp_images"], data['labels']
        assert len(self.images) == len(self.seg_images)
        assert len(self.images) == len(self.labels)
        # print(len(self.images))

        self.img_folder = img_folder

    def __getitem__(self, index):
        image = self.images[index]
        corp_image = self.seg_images[index]
        label = self.labels[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, corp_image, label

    def __len__(self):
        return len(self.images)