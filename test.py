import datasets
import torch
from torchvision import transforms
from train import model
from torch.autograd import Variable

dataset = "/media/disk/lds/dataset/brain_tumor/512+128/1"
test_batch_size = 20

# Converts batches of class indices to classes of one-hot vectors.
def to_one_hot(index, length):
    # batch_size = x.size(0)
    # x_one_hot = torch.zeros(batch_size, length)
    # for i in range(batch_size):
    #     x_one_hot[i, x[i]] = 1.0
    return torch.zeros(test_batch_size, length).scatter_(1, index, 1)

def test(dataset):
    dataset_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.TUMOR_IMG(dataset, train=False, transform=dataset_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)
    # 使用模型，
    model.eval()
    test_loss = 0
    correct = 0

    for batch_idx, (images, corp_images, labels) in enumerate(test_loader):
        target_indices = target
        target_one_hot = to_one_hot(target_indices, length=model.digits.num_units)

        data, target = Variable(images.float()).cuda(), Variable(target_one_hot).cuda()

        output = model(images, corp_images)

        test_loss += model.loss(images, output, target, size_average=False).data # sum up batch loss

        v_mag = torch.sqrt((output**2).sum(dim=2, keepdim=True))

        pred = v_mag.data.max(1, keepdim=True)[1].cpu()

        correct += pred.eq(target_indices.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))