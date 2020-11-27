#
# Dynamic Routing Between Capsules
# https://arxiv.org/pdf/1710.09829.pdf
#

import os

import torch
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms

import datasets
from capsule_network import CapsuleNetwork

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
# Settings.
#

learning_rate = 0.01

batch_size = 20
test_batch_size = 20

# Stop training if loss goes below this threshold.
early_stop_loss = 0.0001
dataset = "/media/disk/lds/dataset/brain_tumor/512+128/1"

# load the data

# Normalization for TUMOR dataset.
dataset_transform = transforms.Compose([
                       transforms.ToTensor(),
                       # transforms.Normalize((0.1307,), (0.3081,))
                   ])

train_dataset = datasets.TUMOR_IMG(dataset, train=True, transform=dataset_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.TUMOR_IMG(dataset, train=False, transform=dataset_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

#
# Create capsule network.
#
conv_inputs = 64
conv_outputs = 256
num_primary_units = 8
primary_unit_size = 24 * 24 * 32  # fixme get from conv2d
output_unit_size = 32

network = CapsuleNetwork(image_width=512,
                         image_height=512,
                         image_channels=1,
                         conv_inputs=conv_inputs,
                         conv_outputs=conv_outputs,
                         num_primary_units=num_primary_units,
                         primary_unit_size=primary_unit_size,
                         num_output_units=3, # one for each MNIST digit
                         output_unit_size=output_unit_size).cuda()
# print(network)


# Converts batches of class indices to classes of one-hot vectors.
def to_one_hot(x, length):
    batch_size = x.size(0)
    x_one_hot = torch.zeros(batch_size, length)
    for i in range(batch_size):
        x_one_hot[i, x[i]] = 1.0
    return x_one_hot

# This is the test function from the basic Pytorch MNIST example, but adapted to use the capsule network.
# https://github.com/pytorch/examples/blob/master/mnist/main.py
def test():
    network.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        target_indices = target
        target_one_hot = to_one_hot(target_indices, length=network.digits.num_units)

        data, target = Variable(data, volatile=True).cuda(), Variable(target_one_hot).cuda()

        output = network(data)

        test_loss += network.loss(data, output, target, size_average=False).data # sum up batch loss

        v_mag = torch.sqrt((output**2).sum(dim=2, keepdim=True))

        pred = v_mag.data.max(1, keepdim=True)[1]

        correct += pred.eq(target_indices.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# This is the train function from the basic Pytorch MNIST example, but adapted to use the capsule network.
# https://github.com/pytorch/examples/blob/master/mnist/main.py
def train(epoch):

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)

    last_loss = None
    log_interval = 1
    network.train()
    for batch_idx, (images, corp_images, labels) in enumerate(train_loader):
        target_one_hot = to_one_hot(labels, length=network.digits.num_units)

        images, corp_images, labels = images.type(torch.FloatTensor).cuda(), \
                                      corp_images.type(torch.FloatTensor).cuda(),\
                                      target_one_hot.cuda()

        optimizer.zero_grad()

        output = network(images, corp_images)

        loss = network.loss(images, output, labels)
        loss.backward()
        last_loss = loss.data

        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(images),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data))

        if last_loss < early_stop_loss:
            break

    return last_loss


num_epochs = 10
for epoch in range(1, num_epochs + 1):
    last_loss = train(epoch)
    # test()
    if last_loss < early_stop_loss:
        break
