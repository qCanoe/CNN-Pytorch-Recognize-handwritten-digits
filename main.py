from __future__ import print_function
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import copy
from tqdm import trange
import numpy as np

from utils.config_utils import read_args, load_config, Dict2Object


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """
    tain the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()
    with open('training_results.txt', 'a') as f:
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            training_loss = loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            training_acc = correct / len(data)
            f.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccuracy: {:.2f}%\n'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), training_loss, 100. * training_acc))
    return training_acc, training_loss


def test(model, device, test_loader):
    """
    test the model and return the tesing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        with open('testing_results.txt', 'a') as f:
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

                testing_loss = test_loss / len(test_loader.dataset)
                testing_acc = 100. * correct / len(test_loader.dataset)
                f.write('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
                    testing_loss, correct, len(test_loader.dataset),
                    testing_acc))
    return testing_acc, testing_loss


def plot(epoches, performance, filename, title, ylabel):
    """
    plot the model peformance
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    import matplotlib.pyplot as plt
    plt.plot(epoches, performance)
    plt.xlabel('Epoches')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename)
    plt.cla()



def run(config):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(config.seed)
    print("使用的seed为：", config.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 3,
                       'pin_memory': True,}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    # 添加随机种子
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)


    # 记录训练和测试的准确率和损失
    epoches = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in trange(1, config.epochs + 1):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        # 记录训练信息
        epoches.append(epoch)
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)
        test_acc, test_loss = test(model, device, test_loader)
        # 记录测试信息
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)
        scheduler.step()

    # 绘制训练和测试的准确率和损失曲线
    plot(epoches, training_accuracies, f'training_accuracies_{config.seed}.png','training accuracy', 'accuracy')
    plot(epoches, training_loss, f'training_loss_{config.seed}.png', 'training loss', 'loss')
    plot(epoches, testing_accuracies, f'testing_accuracies_{config.seed}.png', 'testing accuracy', 'accuracy')
    plot(epoches, testing_loss, f'testing_loss_{config.seed}.png', 'testing loss', 'loss')
    plot_mean(epoches, testing_accuracies, f'mean_accuracy_{config.seed}.png')
    print(f"seed:{config.seed} end.")
    if config.save_model:
        torch.save(model.state_dict(), f"mnist_cnn_{config.seed}.pt")

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_mean(epoches, testing_accuracies, file_name):
    """
    Plot the mean accuracy of each epoch.
    :param epoches: recorded epoches
    :param testing_accuracies: recorded testing accuracies
    :return:
    """
    mean_accuracies = moving_average(testing_accuracies, 3)

    # plot the mean accuracies
    plt.plot(list(range(len(mean_accuracies))), mean_accuracies, 'b-')
    plt.xlabel('Epoches')
    plt.ylabel('Mean Accuracy')
    plt.title('Mean Accuracy of Each Epoch')
    plt.savefig(file_name)
    plt.cla()



if __name__ == '__main__':
    arg = read_args()

    # 加载训练设置
    config = load_config(arg)

    executor = ProcessPoolExecutor(max_workers=3)
    all_task = []
    for seed in config.seeds:
        cur_config = copy.deepcopy(config)
        cur_config.seed = seed
        task = executor.submit(run, (cur_config))
        all_task.append(task)
    wait(all_task, return_when=ALL_COMPLETED)
    print("all end")
    # 训练模型并记录结果
    # run(config)
    