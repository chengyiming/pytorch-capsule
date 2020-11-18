
from data_handler import get_data
import torch
import torch.utils.data


batch_size = 128

def train(dataset, ckpt = None, output = None):
    X_train, seg_train, y_train, X_test, seg_test, y_test = get_data(dataset)

    # 归一化处理
    X_train = X_train / 255
    seg_train = seg_train / 255
    X_test = X_test / 255
    seg_test = seg_test / 255

    # 加载器加载
    X_train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True)
    seg_train_loader = torch.utils.data.DataLoader(seg_train, batch_size=batch_size, shuffle=True)
    y_train_loader = torch.utils.data.DataLoader(y_train, batch_size=batch_size, shuffle=True)
    X_test_loader = torch.utils.data.DataLoader(X_test, batch_size=batch_size, shuffle=True)
    seg_test_loader = torch.utils.data.DataLoader(seg_test, batch_size=batch_size, shuffle=True)
    y_test_loader = torch.utils.data.DataLoader(y_test, batch_size=batch_size, shuffle=True)

