import torch
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms

import datasets
from capsule_network import CapsuleNetwork
import test
from tensorboardX import SummaryWriter
import os
from conf import global_settings as settings

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


learning_rate = 0.0001

batch_size = 50

# Stop training if loss goes below this threshold.
early_stop_loss = 0.0001
dataset = "/media/disk/lds/dataset/brain_tumor/512+128/1"

conv_inputs = 64
conv_outputs = 256
num_primary_units = 8
primary_unit_size = 24 * 24 * 32  # fixme get from conv2d
output_unit_size = 32



MAX_EPOCH = 50
start_epoch = 0
checkpoint_file = 'model.pt'

# Converts batches of class indices to classes of one-hot vectors.
def to_one_hot(batch_size, index, length):
    return torch.zeros(batch_size, length).scatter(1, index.unsqueeze(1).long(), 1)
model = CapsuleNetwork(image_width=512,
                             image_height=512,
                             image_channels=1,
                             conv_inputs=conv_inputs,
                             conv_outputs=conv_outputs,
                             num_primary_units=num_primary_units,
                             primary_unit_size=primary_unit_size,
                             num_output_units=3,  # one for each MNIST digit
                             output_unit_size=output_unit_size).cuda()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Normalization for TUMOR dataset.
# ToTensor将图片从(N W H C)->(N C W H)
def train(dataset, model, optimizer, start_epoch, output_path = None):
    dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.TUMOR_IMG(dataset, train=True, transform=dataset_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    # 尝试加载
    for epoch in range(start_epoch + 1, MAX_EPOCH):
        last_loss = None
        log_interval = 1
        model.train()
        for batch_idx, (images, corp_images, labels) in enumerate(train_loader):
            # images = images.transpose(1,3).transpose(2,3)
            # corp_images = corp_images.transpose(1,3).transpose(2,3)
            # print("images[0]:", images[0])
            origin_labels = labels.long().cuda()
            # images.type(): torch.DoubleTensor
            # images.size(): torch.Size([20, 1, 512, 512])
            # labels.type(): torch.IntTensor
            # labels.size(): torch.Size([20])
            # print("images-max-min:", torch.max(images[0][0][0][100]), torch.min(images))

            target_one_hot = to_one_hot(images.size(0), labels, length=model.digits.num_units)

            images, corp_images, labels = images.float().cuda(), \
                                          corp_images.float().cuda(), \
                                          target_one_hot.cuda()

            optimizer.zero_grad()

            output = model(images, corp_images)

            # 总的迭代次数
            n_iter  = (epoch - 1)*len(train_loader) + batch_idx - 1


            loss = model.loss(images, output, labels)
            _, _, acc = model.acc(output, origin_labels)
            loss.backward()
            last_loss = loss.data
            optimizer.step()

            # 以每个batch为单位的日志记录
            writer.add_scalar("Train/acc(batch)", acc, n_iter)
            writer.add_scalar("Train/loss(batch)", loss.item(), n_iter)

            # TODO 不知道是干啥的，观察观察
            for name, param in model.named_parameters():
                layer, attr = os.path.splitext(name)
                attr = attr[1:]
                writer.add_histogram("{}/{}".format(layer, attr), param, epoch)


            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc:{:.6f}'.format(
                    epoch,
                    batch_idx * len(images),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.data,
                    acc.data
                    ))

            if last_loss < early_stop_loss:
                break

        # 保存
        if not os.path.exists(output_path):
            # 创建目录
            os.makedirs(output_path)
        test.test(dataset, epoch)
        checkpoint = {
            'model_state_dict':model.state_dict(),
            'optimizer.state_dict':optimizer.state_dict(),
            'epoch':epoch
        }
        torch.save(checkpoint, os.path.join(output_path, checkpoint_file))


if __name__ == "__main__":
    checkpoint = None
    output_path = "./outputs"
    if os.path.exists(output_path):
        checkpoint = torch.load(os.path.join(output_path, checkpoint_file))
    if checkpoint != None:
        print("load from ckpt")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer.state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        print("train from the begining")
    writer = SummaryWriter(logdir=os.path.join(settings.LOGDIR, settings.TIME_NOW))
    train(dataset, model, optimizer, start_epoch, output_path)
    writer.close()