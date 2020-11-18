import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms

import datasets
dataset = "/media/disk/lds/dataset/brain_tumor/512+128/1"
dataset_transform = transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])

train_dataset = datasets.TUMOR_IMG(dataset,train=True, transform = dataset_transform)
test_dataset = datasets.TUMOR_IMG(dataset,train=False, transform= dataset_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=6, shuffle=True)



def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

for step ,(b_x, b_seg, b_y) in enumerate(train_loader):
    print("step:", step)

    if step < 1:
        # imgs = torchvision.utils.make_grid(b_x)
        # print(imgs.shape)
        # imgs = np.transpose(imgs,(1,2,0))
        # print(imgs.shape)
        # plt.imshow(imgs)
        # plt.show()
        print(len(b_x))
        print(len(b_x[0]))
        print(len(b_x[0][0]))
        print(len(b_x[0][0][0]))

        break