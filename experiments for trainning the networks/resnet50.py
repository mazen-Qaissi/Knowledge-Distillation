import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import cv2
torch.cuda.empty_cache()
writer = SummaryWriter()
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
# Hyper-parameters
num_epochs = 50
learning_rate = 0.001

# Image preprocessing modules
# transform = transforms.Compose([
#     transforms.Pad(4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32),
#     transforms.ToTensor()])
# Normalize training set together with augmentation
# transform_train = transforms.Compose([
#     transforms.Pad(4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomResizedCrop(64),
#     transforms.RandomCrop(32),
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[x / 255.0 for x in [0.507, 0.487, 0.441]],
#                          std=[x / 255.0 for x in [0.267, 0.256, 0.276]])
# ])
#
# # Normalize test set same as training set without augmentation
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Resize(64),
#     transforms.Normalize(mean=[x / 255.0 for x in [0.507, 0.487, 0.441]],
#                          std=[x / 255.0 for x in [0.267, 0.256, 0.276]])
#####################################3
# trans = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor(),
#          transforms.Normalize(mean=[x / 255.0 for x in [0.507, 0.487, 0.441]],
#                               std=[x / 255.0 for x in [0.267, 0.256, 0.276]])]
# trans = transforms.Compose(trans)
# transT = [transforms.ToTensor(),
#           transforms.Normalize(mean=[x / 255.0 for x in [0.507, 0.487, 0.441]],
#                                std=[x / 255.0 for x in [0.267, 0.256, 0.276]])]
# transT = transforms.Compose(transT)
# --------------------------------------------------------------------------------------------
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])
# ------------------------------------------------------------------------------------------------
# CIFAR-100 dataset
train_dataset = torchvision.datasets.CIFAR100(root='../../data/', train=True, transform=transform_train, download=True)

test_dataset = torchvision.datasets.CIFAR100(root='../../data/', train=False, transform=transform_test)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)
###############
##############
#
# class ResidualBlock(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
#         super(ResidualBlock, self).__init__()
#
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1)
#         self.batch_norm1 = nn.BatchNorm2d(out_channels)
#
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1)
#         self.batch_norm2 = nn.BatchNorm2d(out_channels)
#
#         self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1)
#         self.batch_norm3 = nn.BatchNorm2d(out_channels * self.expansion)
#
#         self.i_downsample = i_downsample
#         self.stride = stride
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         identity = x.clone()
#
#         x = self.relu(self.batch_norm1(self.conv1(x)))
#         # print("conv1:",x.shape)
#         x = self.relu(self.batch_norm2(self.conv2(x)))
#         # print("conv2:",x.shape)
#
#         x = self.conv3(x)
#         # print("conv3:",x.shape)
#
#         x = self.batch_norm3(x)
#         # print("batch_conv3:",x.shape)
#
#         # downsample if needed
#         if self.i_downsample is not None:
#             identity = self.i_downsample(identity)
#             # print("identity:",identity.shape)
#
#         # add identity
#         # print(identity.size())
#
#         x += identity
#         # print(x.size())
#         # print("x+identity:",x.shape)
#
#         x = self.relu(x)
#         # print("relu:",x.shape)
#
#         return x
#
#
# class ResNet(nn.Module):
#     def __init__(self, ResBlock, layer_list, num_classes, num_channels=3):
#         super(ResNet, self).__init__()
#         self.in_channels = 64
#         self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=3, stride=1)
#         self.batch_norm1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU()
#         # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
#         self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=64)
#         self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=128)
#         self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=256)
#         self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=512)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc1 = nn.Linear(512 * ResBlock.expansion, 256)
#         self.fc2 = nn.Linear(256, num_classes)
#
#     def forward(self, x):
#         x = self.relu(self.batch_norm1(self.conv1(x)))
#         # print("conv1:",x.shape)
#         # x = self.max_pool(x)
#         x = self.layer1(x)
#         # print("layer1:",x.shape)
#
#         x = self.layer2(x)
#         # print("layer2:",x.shape)
#
#         x = self.layer3(x)
#         # print("layer3:",x.shape)
#
#         x = self.layer4(x)
#         # print("layer4:",x.shape)
#
#         x = self.avgpool(x)
#         # print("avgpool:",x.shape)
#
#         x = x.reshape(x.shape[0], -1)
#         # print("flatten:",x.shape)
#
#         x = self.fc1(x)
#         # print("fc1:",x.shape)
#
#         x = self.fc2(x)
#         # print("fc2:",x.shape)
#
#         return x
#
#     def _make_layer(self, ResBlock, blocks, planes, stride=1):
#         ii_downsample = None
#         layers = []
#         if stride != 1 or self.in_channels != planes * ResBlock.expansion:
#             ii_downsample = nn.Sequential(
#                 nn.Conv2d(self.in_channels, planes * ResBlock.expansion, kernel_size=1, stride=stride),
#                 nn.BatchNorm2d(planes * ResBlock.expansion)
#             )
#
#         layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
#         self.in_channels = planes * ResBlock.expansion
#
#         for i in range(blocks - 1):
#             layers.append(ResBlock(self.in_channels, planes))
#
#         return nn.Sequential(*layers)
########################
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.LeakyReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU()(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU())
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = self.dropout(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


########################

model = ResNet(BottleNeck, [3, 4, 6, 3]).to(device)
print(device)
count = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("count:", count)
# ------------------------------------------------------------------------------------
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# For updating learning rate
def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Train the model
loss_arr = []
total_step = len(train_loader)
curr_lr = learning_rate

for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = ((outputs.argmax(dim=1) == labels).float().mean())
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

    print('Epoch : {}, train accuracy : {}, train loss : {}'.format(epoch + 1, epoch_accuracy, epoch_loss))
    writer.add_scalar("Loss/train", epoch_loss, (epoch + 1))
    writer.add_scalar("accuracy/train", epoch_accuracy, (epoch + 1))
    loss_arr.append(loss.item())

    # Decay learning rate
    if (epoch + 1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)
writer.flush()
# Test the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the model on the train images: {} %'.format(100 * correct / total))
writer.close()

# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')
