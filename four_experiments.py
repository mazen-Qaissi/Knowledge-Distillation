import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import ResNet18CIFAR10
import ResNet18CIFAR10Transforms

num_epochs = 15
learning_rate = 0.002

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


print("student")
logits_student = ResNet18CIFAR10.run_resnet18_student()
print("teacher")
model_teacher = ResNet18CIFAR10Transforms.run_resnet18_teacher()


writer1 = SummaryWriter()
#
Temp_step = [3, 3.5, 4, 4.5, 5]
# # trying different Temps
# print("the first experiment is trying different T with distillation")
# for t in Temp_step:
#     model_1 = ResNet(ResidualBlock, [2, 2, 2, 2]).to("cuda")
#     count = sum(p.numel() for p in model_1.parameters() if p.requires_grad)
#     print("count:", count)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)
#     # Training loop
#     print("Temperature num is:", t)
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         running_acc = 0.0
#         for data, target in train_loader:
#             images = data.to("cuda")
#             labels = target.to("cuda")
#             optimizer.zero_grad()
#             output = model_1(images)
#             loss1 = criterion(output, labels)
#             loss = loss1 + dist_loss(model_teacher(images).detach(), output, t)
#             running_loss += loss.item()
#             _, predicted = torch.max(output.data, 1)
#             running_acc += (predicted == labels).sum().item()
#             loss.backward()
#             optimizer.step()
#         # Print the average loss and accuracy for the epoch
#         print(f'Epoch {epoch + 1} Train Loss: {running_loss / len(train_loader)} Train Acc: {(running_acc / len(train_dataset)) * 100}')
#         writer1.add_scalar("distillation/Loss/train/1", running_loss / len(train_loader), (epoch + 1))
#         writer1.add_scalar("distillation/Acc/train/1", (running_acc / len(train_dataset)) * 100, (epoch + 1))
#         # Testing loop
#         val_loss = 0.0
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for inputs, labels in test_loader:
#                 inputs, labels = inputs.cuda(), labels.cuda()
#                 outputs = model_1(inputs)
#                 loss1 = F.cross_entropy(outputs, labels)
#                 teacher_p = model_teacher(inputs).detach()
#                 loss = loss1 + dist_loss(teacher_p, outputs, t)
#                 val_loss += loss.item()
#                 _, pred = torch.max(outputs, 1)
#                 total += labels.size(0)
#                 correct += (pred == labels).sum().item()
#             val_acc = (correct / total) * 100
#             print('Epoch {}:, test loss {}, test acc {}'.format(epoch + 1, val_loss / len(test_loader), val_acc))
#             writer1.add_scalar("distillation/Loss/test/1", val_loss / len(test_loader), (epoch + 1))
#             writer1.add_scalar("distillation/Acc/test/1", val_acc, (epoch + 1))
#     writer1.flush()
#     model_1.eval()
#     with torch.no_grad():
#         correct = 0
#         total = 0
#         for images, labels in test_loader:
#             images = images.to("cuda")
#             labels = labels.to("cuda")
#             outputs = model_1(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#         print('Accuracy of the model of distillation on the test images: {} %'.format(100 * correct / total))
#         for images, labels in train_loader:
#             images = images.to("cuda")
#             labels = labels.to("cuda")
#             outputs = model_1(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#         print('Accuracy of the model of distillation on the train images: {} %'.format(100 * correct / total))
#     writer1.close()

# trying different Temps in one run but after some epochs we changed the Temp
temp_epoch = [[1,2,3,4,5],[2,2.5,3,3.5,4],[2.5,3,3.5,4,4.5],[1.5,2.5,3.5,4.5,5.5]]
print("The second experiment is put different T after some epoch with distillation")
model_2 = ResNet(ResidualBlock, [2, 2, 2, 2]).to("cuda")
count = sum(p.numel() for p in model_2.parameters() if p.requires_grad)
print("count:", count)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_2.parameters(), lr=learning_rate)
# Training loop
for number in range(1):
    number = 2
    print("Temperature set is:", temp_epoch[number])
    k = 0
    t = temp_epoch[number][k]
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_acc = 0.0
        for data, target in train_loader:
            images = data.to("cuda")
            labels = target.to("cuda")
            optimizer.zero_grad()
            output = model_2(images)
            loss1 = criterion(output, labels)
            loss = loss1 + dist_loss(model_teacher(images).detach(), output, t)
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            running_acc += (predicted == labels).sum().item()
            loss.backward()
            optimizer.step()
        # Print the average loss and accuracy for the epoch
        print(f'Epoch {epoch + 1} Train Loss: {running_loss / len(train_loader)} Train Acc: {(running_acc / len(train_dataset)) * 100}')
        writer1.add_scalar("distillation/Loss/train/2", running_loss / len(train_loader), (epoch + 1))
        writer1.add_scalar("distillation/Acc/train/2", (running_acc / len(train_dataset)) * 100, (epoch + 1))
        # Testing loop
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()
                outputs = model_2(inputs)
                loss1 = F.cross_entropy(outputs, labels)
                teacher_p = model_teacher(inputs).detach()
                loss = loss1 + dist_loss(teacher_p, outputs, t)
                val_loss += loss.item()
                _, pred = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
            val_acc = (correct / total) * 100
            print('Epoch {}:, test loss {}, test acc {}'.format(epoch + 1, val_loss / len(test_loader), val_acc))
            writer1.add_scalar("distillation/Loss/test/2", val_loss / len(test_loader), (epoch + 1))
            writer1.add_scalar("distillation/Acc/test/2", val_acc, (epoch + 1))
        if (epoch%6==0):
            t = temp_epoch[number][k]
            k += 1

    writer1.flush()
    model_2.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to("cuda")
            labels = labels.to("cuda")
            outputs = model_2(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Accuracy of the model of distillation on the test images: {} %'.format(100 * correct / total))
        for images, labels in train_loader:
            images = images.to("cuda")
            labels = labels.to("cuda")
            outputs = model_2(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy of the model of distillation on the train images: {} %'.format(100 * correct / total))
    writer1.close()

# trying different alfa in function distillation
alfa = [0,0.1,0.25,0.5,0.75,0.9]
print("The third experiment is trying different alfa in function distillation")
for a in alfa:
    for t in Temp_step:
        model_1 = ResNet(ResidualBlock, [2, 2, 2, 2]).to("cuda")
        count = sum(p.numel() for p in model_1.parameters() if p.requires_grad)
        print("count:", count)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)
        # Training loop
        print("Temperature num is:", t)
        print("Alfa is:", a)
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_acc = 0.0
            for data, target in train_loader:
                images = data.to("cuda")
                labels = target.to("cuda")
                optimizer.zero_grad()
                output = model_1(images)
                loss1 = criterion(output, labels)
                loss = (a * loss1) + ((1-a) * dist_loss(model_teacher(images).detach(), output, t))
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                running_acc += (predicted == labels).sum().item()
                loss.backward()
                optimizer.step()
            # Print the average loss and accuracy for the epoch
            print(f'Epoch {epoch + 1} Train Loss: {running_loss / len(train_loader)} Train Acc: {(running_acc / len(train_dataset)) * 100}')
            writer1.add_scalar("distillation/Loss/train/3", running_loss / len(train_loader), (epoch + 1))
            writer1.add_scalar("distillation/Acc/train/3", (running_acc / len(train_dataset)) * 100, (epoch + 1))
            # Testing loop
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model_1(inputs)
                    loss1 = F.cross_entropy(outputs, labels)
                    teacher_p = model_teacher(inputs).detach()
                    loss = (a * loss1) + ((1-a) * dist_loss(model_teacher(images).detach(), output, t))
                    val_loss += loss.item()
                    _, pred = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
                val_acc = (correct / total) * 100
                print('Epoch {}:, test loss {}, test acc {}'.format(epoch + 1, val_loss / len(test_loader), val_acc))
                writer1.add_scalar("distillation/Loss/test/3", val_loss / len(test_loader), (epoch + 1))
                writer1.add_scalar("distillation/Acc/test/3", val_acc, (epoch + 1))
        writer1.flush()
        model_1.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model_1(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the model of distillation on the test images: {} %'.format(100 * correct / total))
            for images, labels in train_loader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model_1(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of the model of distillation on the train images: {} %'.format(100 * correct / total))
        writer1.close()

