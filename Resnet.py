# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import torch
from torchvision import transforms
from torchvision import datasets
import torch.nn as nn
import os
from pathlib import Path
from tqdm import tqdm

path = os.getcwd()
datapath = str(Path(__file__).parents[1])

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
class ResBlock(nn.Module):
    def __init__(self, inc: int, outc: int, stride=1):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU()
        self.inc = inc
        self.outc = outc
        self.stride = stride
        self.block = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride,bias=False),
            nn.BatchNorm2d(outc),
            nn.ReLU(),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=1,bias=False),
            nn.BatchNorm2d(outc),
        )
        # if out and in channels are not equal, change the size of the input to allow addtion

        self.extra = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=1, stride=stride,bias=False),
            nn.BatchNorm2d(outc),
        )


    def forward(self, x: torch.Tensor):
        identity = x
        if self.inc != self.outc or self.stride != 1:
            identity = self.extra(x)
        out = self.block(x) + identity
        return self.relu(out)


class Resnet(nn.Module):
    def __init__(self, block, num_blocks, num_class: int = 10):
        super(Resnet, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.in_channels = 64
        self.flatten = nn.Flatten()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.MaxPool2d(3, stride=2,padding=1)
        )
        self.layer1 = self.make_layer(block, 64, 1, num_blocks[0])
        self.layer2 = self.make_layer(block, 128, 2, num_blocks[1])
        self.layer3 = self.make_layer(block, 256, 2, num_blocks[2])
        self.layer4 = self.make_layer(block, 512, 2, num_blocks[3])
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(512, num_class)
        self.softmax = nn.Softmax(dim=1)

    def make_layer(self, block, outc, stride, num_blocks):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, outc, stride))
            self.in_channels = outc
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        x = self.input_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg(x)
        x = self.flatten(x)
        x = self.linear(x)
        # x = self.softmax(x)
        return x


if __name__ == '__main__':
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print("running on ", device)
    cifar_train = datasets.CIFAR10(datapath + '/cifar10',
                                   True,
                                   transform=transform_train,
                                   download=True)
    cifar_test = datasets.CIFAR10(datapath + '/cifar10',
                                  False,
                                  transform=transform_test,
                                  download=True)
    training_loader = torch.utils.data.DataLoader(cifar_train, batch_size=32, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(cifar_test, batch_size=32, shuffle=False)

    #model = Resnet(ResBlock, [2, 2, 2, 2], 10).to(device) #18
    model = Resnet(ResBlock, [3, 4, 6, 3], 10).to(device) #34
    #model = Resnet(ResBlock, [3, 4, 6, 3], 10).to(device) #50
    #model = Resnet(ResBlock, [3, 4, 6, 3], 10).to(device) #101
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 100


    def valid():

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for x, label in validation_loader:
                x, label = x.to(device), label.to(device)
                logits = model.forward(x)
                pred = logits.argmax(dim=1)
                correct += torch.eq(pred, label).float().sum().item()
                total += x.shape[0]
            print('score:', correct / total)


    torch.cuda.synchronize()
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    for epoch in (range(epochs)):
        model.train()
        #optimizer.param_groups[0]['lr'] = 10 ** (-2 - (epoch / 40))
        running_loss = 0.0
        correct = 0
        total = 0
        pbar = tqdm(total=len(training_loader), position=0, leave=True)
        for id, (x, label) in enumerate(training_loader):
            x, label = x.to(device), label.to(device)
            # pass the input through the CNN
            logits = model.forward(x)
            # calculate the loss
            loss = model.loss(logits, label)
            # reset the gradient of the loss to 0 for this batch
            optimizer.zero_grad()
            # do backward propagation for the current batch
            loss.backward()
            # update the parameters from the optimizer
            optimizer.step()
            running_loss += loss.item()
            pbar.update(1)
        print(epoch, 'loss:', loss.item())
        valid()
    torch.save(model.state_dict(), path + '/ClassicResnet34-3')
