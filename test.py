import datasets
import torch
from torchvision import transforms
from tensorboardX import SummaryWriter
import conf.global_settings as settings
import os
from capsule_network import CapsuleNetwork


conv_inputs = 64
conv_outputs = 256
num_primary_units = 8
primary_unit_size = 24 * 24 * 32  # fixme get from conv2d
output_unit_size = 32


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

writer = SummaryWriter(logdir=os.path.join(settings.LOGDIR, settings.TIME_NOW))

dataset = "/media/disk/lds/dataset/brain_tumor/512+128/1"
test_batch_size = 32

# Converts batches of class indices to classes of one-hot vectors.
def to_one_hot(batch_size, index, length):
    return torch.zeros(batch_size, length).scatter(1, index.unsqueeze(1).long(), 1)

def test(dataset, epoch):
    dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.TUMOR_IMG(dataset, train=False, transform=dataset_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    model.eval()
    test_loss = 0
    correct = 0

    for batch_idx, (images, corp_images, target) in enumerate(test_loader):

        batch_size = images.size(0)

        target_indices = target.long()
        target_one_hot = to_one_hot(batch_size, target, length=model.digits.num_units)

        images, corp_images, target = images.float().cuda(), \
                                    corp_images.float().cuda(),\
                                    target_one_hot.cuda()

        output = model(images, corp_images)

        test_loss += model.loss(images, output, target, size_average=False).data.sum(dim=0) # sum up batch loss

        v_mag = torch.sqrt((output**2).sum(dim=2, keepdim=True))

        pred = v_mag.data.max(1, keepdim=True)[1]

        correct += pred.eq(target_indices.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    # 日志记录
    writer.add_scalar("Test/acc", correct / len(test_loader.dataset), epoch)
    writer.add_scalar("Test/loss", test_loss, epoch)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == "__main__":
    test(dataset, 3)