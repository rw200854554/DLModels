# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from pathlib import Path
import os
from tqdm import tqdm
path = os.getcwd()
datapath =  str(Path(__file__).parents[1])
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
class VGGnet(nn.Module):
    def __init__(self,num_class:int=10):
        super().__init__()
        self.loss = nn.CrossEntropyLoss();
        self.feature = nn.Sequential(

            nn.Conv2d(3,64,stride=1,padding=1,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,64,stride=1,padding=1,kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2,padding=0),

            nn.Conv2d(64,128,stride=1,padding=1,kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2,padding=0),

            nn.Conv2d(128, 256, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2,padding=0),

            nn.Conv2d(256, 512, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2,padding=0),

            nn.Conv2d(512, 512, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, stride=1, padding=1, kernel_size=3),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2,padding=0),

            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(25088,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,num_class),
            #nn.Softmax(dim=1),
            #nn.Softmax(dim=1)

        )
    def forward(self, x: torch.Tensor ):
        x = self.feature(x)
        return x

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    cifar_train = datasets.CIFAR10(datapath + '/cifar10',
                                   True,
                                   transform=transform,
                                   download=True)
    cifar_test = datasets.CIFAR10(datapath + '/cifar10',
                                  False,
                                  transform=transform,
                                  download=True)
    training_loader = torch.utils.data.DataLoader(cifar_train, batch_size=32, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(cifar_test, batch_size=32, shuffle=False)


    model = VGGnet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01 )

    epochs = 100

    for epoch in (range(epochs)):
        model.train()
        for id, (x, label) in tqdm(enumerate(training_loader)):
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
        print(epoch,'loss:', loss.item())
    torch.save(model.state_dict(), path+'/ClassicVGG')
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