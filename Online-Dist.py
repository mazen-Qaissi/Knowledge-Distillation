import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import ResNet18CIFAR10
import ResNet18CIFAR10Transforms

num_epochs = 20
learning_rate = 0.0001


# CIFAR-10 dataset Teacher

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

train_dataset_T = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

test_dataset_T = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# Data loader
train_loader_T = torch.utils.data.DataLoader(dataset=train_dataset_T, batch_size=256, shuffle=True)

test_loader_T = torch.utils.data.DataLoader(dataset=test_dataset_T, batch_size=256, shuffle=False)


# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='./dataS', train=True, download=True, transform=transforms.ToTensor())

test_dataset = torchvision.datasets.CIFAR10(root='./dataS', train=False, download=True, transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv = conv3x3(3, 64)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.layer4 = self.make_layer(block, 512, layers[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def dist_loss(t, s, T):
    prob_t = F.softmax(t / T, dim=1)
    log_prob_s = F.log_softmax(s / T, dim=1)
    dist_loss = -(prob_t * log_prob_s).sum(dim=1).mean()
    return dist_loss


writer1 = SummaryWriter()

model_T = ResNet(ResidualBlock, [2, 2, 2, 2]).to("cuda")
count = sum(p.numel() for p in model_T.parameters() if p.requires_grad)
print(count)
criterion_T = nn.CrossEntropyLoss()
optimizer_T = torch.optim.Adam(model_T.parameters(), lr=learning_rate)
model_S = ResNet(ResidualBlock, [2, 2, 2, 2]).to("cuda")
count = sum(p.numel() for p in model_S.parameters() if p.requires_grad)
print("count:", count)
# Loss and optimizer
criterion_S = nn.CrossEntropyLoss()
optimizer_S = torch.optim.Adam(model_S.parameters(), lr=learning_rate)

model = ResNet(ResidualBlock, [2, 2, 2, 2]).to("cuda")
count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("count:", count)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    # teacher
    running_loss_T = 0.0
    running_acc_T = 0.0
    for i, data in enumerate(train_loader_T):
        inputs, labels = data
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")
        optimizer_T.zero_grad()
        outputs_T = model_T(inputs)
        loss_T = criterion_T(outputs_T, labels)
        running_loss_T += loss_T.item()
        _, predicted = torch.max(outputs_T.data, 1)
        running_acc_T += (predicted == labels).sum().item()
        loss_T.backward()
        optimizer_T.step()
    # Print the average loss and accuracy for the epoch
    print(f'Teacher Epoch {epoch + 1} Loss: {running_loss_T / len(train_loader_T)} Acc: {running_acc_T / len(train_dataset_T)}')
    writer1.add_scalar("teacher/Loss/train", running_loss_T / len(train_loader_T), (epoch + 1))
    writer1.add_scalar("teacher/Acc/train", (running_acc_T / len(train_dataset_T)) * 100, (epoch + 1))
    # Testing loop
    with torch.no_grad():
        test_loss_T = 0.0
        test_acc_T = 0.0
        for data in test_loader_T:
            inputs, labels = data
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            outputs = model_T(inputs)
            test_loss_T += criterion_T(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            test_acc_T += (predicted == labels).sum().item()

    # Print the average loss and accuracy for the test set
    print(f'Teacher Test Loss: {test_loss_T / len(test_loader_T)} Test Acc: {(test_acc_T / len(test_dataset_T)) * 100}')
    writer1.add_scalar("teacher/Loss/test", test_loss_T / len(test_loader_T), (epoch + 1))
    writer1.add_scalar("teacher/Acc/test", (test_acc_T / len(test_dataset_T)) * 100, (epoch + 1))
    # student
    running_loss_S = 0.0
    running_acc_S = 0.0
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.to("cuda")
        labels = labels.to("cuda")
        optimizer_S.zero_grad()
        outputs_S = model_S(inputs)
        loss_S = criterion_S(outputs_S, labels)
        running_loss_S += loss_S.item()
        _, predicted = torch.max(outputs_S.data, 1)
        running_acc_S += (predicted == labels).sum().item()
        loss_S.backward()
        optimizer_S.step()

    # Print the average loss and accuracy for the epoch
    print(f'Student Epoch {epoch + 1} Loss: {running_loss_S / len(train_loader)} Acc: {(running_acc_S / len(train_dataset)) * 100}')
    writer1.add_scalar("student/Loss/train", running_loss_S / len(train_loader), (epoch + 1))
    writer1.add_scalar("student/Acc/train", (running_acc_S / len(train_dataset)) * 100, (epoch + 1))
    logits = outputs_S
    # Testing loop
    with torch.no_grad():
        test_loss_S = 0.0
        test_acc_S = 0.0
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            outputs = model_S(inputs)
            test_loss_S += criterion_S(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            test_acc_S += (predicted == labels).sum().item()

    # Print the average loss and accuracy for the test set
    print(f' Student Test Loss: {test_loss_S / len(test_loader)} Test Acc: {(test_acc_S / len(test_dataset)) * 100}')
    writer1.add_scalar("student/Loss/test", test_loss_S / len(test_loader), (epoch + 1))
    writer1.add_scalar("student/Acc/test", (test_acc_S / len(test_dataset)) * 100, (epoch + 1))

    # distillation
    running_loss = 0.0
    running_acc = 0.0
    for data, target in train_loader:
        images = data.to("cuda")
        labels = target.to("cuda")
        optimizer.zero_grad()
        output = model(images)
        loss1 = criterion(output, labels)
        loss = loss1 + dist_loss(model_T(images).detach(), output, 4)
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        running_acc += (predicted == labels).sum().item()
        loss.backward()
        optimizer.step()
    # Print the average loss and accuracy for the epoch
    print(f'Distillation Epoch {epoch + 1} Train Loss: {running_loss / len(train_loader)} Train Acc: {(running_acc / len(train_dataset)) * 100}')
    writer1.add_scalar("distillation/Loss/train", running_loss / len(train_loader), (epoch + 1))
    writer1.add_scalar("distillation/Acc/train", (running_acc / len(train_dataset)) * 100, (epoch + 1))
    # Testing loop
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            loss1 = F.cross_entropy(outputs, labels)
            teacher_p = model_T(inputs).detach()
            loss = loss1 + dist_loss(teacher_p, outputs, 4)
            val_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        val_acc = (correct / total) * 100
        print('Distillation Epoch {}:, test loss {}, test acc {}'.format(epoch + 1, val_loss / len(test_loader), val_acc))
        writer1.add_scalar("distillation/Loss/test", val_loss / len(test_loader), (epoch + 1))
        writer1.add_scalar("distillation/Acc/test", val_acc, (epoch + 1))
writer1.flush()
# -------------------------------------------------------------------------------------------------------------------------------------------------------
# model_T.eval()
# with torch.no_grad():
#     test_loss_T = 0.0
#     test_acc_T = 0.0
#     for data in test_loader_T:
#         inputs, labels = data
#         inputs = inputs.to("cuda")
#         labels = labels.to("cuda")
#         outputs = model_T(inputs)
#         test_loss_T += criterion_T(outputs, labels).item()
#         _, predicted = torch.max(outputs.data, 1)
#         test_acc_T += (predicted == labels).sum().item()
#
# # Print the average loss and accuracy for the test set
# print(f'Teacher Test Loss: {test_loss_T / len(test_loader_T)} Test Acc: {(test_acc_T / len(test_dataset_T)) * 100}')
#
# # --------------------------------------------------------------------------------------------------------------------------------------------------------------
# model_S.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to("cuda")
#         labels = labels.to("cuda")
#         outputs_S = model_S(images)
#         _, predicted = torch.max(outputs_S.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print('Accuracy of the model of student on the test images: {} %'.format(100 * correct / total))
#     for images, labels in train_loader:
#         images = images.to("cuda")
#         labels = labels.to("cuda")
#         outputs = model_S(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     print('Accuracy of the model of student on the train images: {} %'.format(100 * correct / total))
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to("cuda")
#         labels = labels.to("cuda")
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print('Accuracy of the model of distillation on the test images: {} %'.format(100 * correct / total))
#     for images, labels in train_loader:
#         images = images.to("cuda")
#         labels = labels.to("cuda")
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     print('Accuracy of the model of distillation on the train images: {} %'.format(100 * correct / total))
writer1.close()
# for epoch in range(num_epochs):
#     for data, target in train_loader:
#         images = data.to("cuda")
#         labels = target.to("cuda")
#         optimizer.zero_grad()
#         output = model(images)
#         loss1 = criterion(output, labels)
#         loss = loss1 + dist_loss(model_teacher(images).detach(), output, 3)
#         loss.backward()
#         optimizer.step()
#
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to("cuda")
#         labels = labels.to("cuda")
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print('Accuracy of the model of distillation on the test images: {} %'.format(100 * correct / total))
#     for images, labels in train_loader:
#         images = images.to("cuda")
#         labels = labels.to("cuda")
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#     print('Accuracy of the model of distillation on the train images: {} %'.format(100 * correct / total))
