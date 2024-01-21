import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
import os
path = os.getcwd()
datapath =  Path(__file__).parents[1]
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((227,227))])
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")
class Alexnet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout:float=0.5 )->None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.features = nn.Sequential(
            nn.Conv2d(3,96,stride=4,padding=0,kernel_size=11),
            nn.ReLU(),
            nn.MaxPool2d(3,stride=2),
            #nn.Conv2d(96,96,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(96,256,kernel_size=5,padding=2,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            #nn.Conv2d(256,256,kernel_size=3,stride=2),
            nn.ReLU(),
            nn.Conv2d(256,384,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(384,384,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            #nn.Conv2d(256,256,kernel_size=3,stride=2),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(9216,4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,num_classes),
        )
    def forward(self, x:torch.Tensor):
        x = self.features(x)
        return x



if __name__ =="__main__":
    cifar_train = datasets.CIFAR10(datapath+'/cifar10',
                                   True,
                                   transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((227, 227)) ]),
                                   download=True)
    cifar_test = datasets.CIFAR10(datapath+ '/cifar10',
                                  False,
                                  transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((227, 227))]),
                                  download=True)
    training_loader = torch.utils.data.DataLoader(cifar_train, batch_size=32, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(cifar_test, batch_size=32, shuffle=False)



    model = Alexnet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01 )

    epochs = 100

    for epoch in range(epochs):
        model.train()
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
        print(id,'loss:', loss.item())
    torch.save(model.state_dict(), path+'/full_conv_Alex')
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