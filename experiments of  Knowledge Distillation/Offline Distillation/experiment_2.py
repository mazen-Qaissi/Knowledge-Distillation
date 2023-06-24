from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import requests
import io
import PIL

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


def get_weights(bit_variant):
    response = requests.get(f'https://storage.googleapis.com/bit_models/{bit_variant}.npz')
    response.raise_for_status()
    return np.load(io.BytesIO(response.content))


class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


# 3x3 convolution
def conv3x3_1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3_1(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_1(out_channels, out_channels)
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
        self.conv = conv3x3_1(3, 64)
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


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


def tf2th(conv_weights):
    """Possibly convert HWIO to OIHW"""
    if conv_weights.ndim == 4:
        conv_weights = np.transpose(conv_weights, [3, 2, 0, 1])
    return torch.from_numpy(conv_weights)


class PreActBottleneck(nn.Module):
    """
    Follows the implementation of "Identity Mappings in Deep Residual Networks" here:
    https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

    Except it puts the stride on 3x3 conv when available.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4

        self.gn1 = nn.GroupNorm(32, cin)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid)
        self.conv2 = conv3x3(cmid, cmid, stride)  # Original ResNetv2 has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride)

    def forward(self, x):
        # Conv'ed branch
        out = self.relu(self.gn1(x))

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(out)

        # The first block has already applied pre-act before splitting, see Appendix.
        out = self.conv1(out)
        out = self.conv2(self.relu(self.gn2(out)))
        out = self.conv3(self.relu(self.gn3(out)))

        return out + residual

    def load_from(self, weights, prefix=''):
        with torch.no_grad():
            self.conv1.weight.copy_(tf2th(weights[prefix + 'a/standardized_conv2d/kernel']))
            self.conv2.weight.copy_(tf2th(weights[prefix + 'b/standardized_conv2d/kernel']))
            self.conv3.weight.copy_(tf2th(weights[prefix + 'c/standardized_conv2d/kernel']))
            self.gn1.weight.copy_(tf2th(weights[prefix + 'a/group_norm/gamma']))
            self.gn2.weight.copy_(tf2th(weights[prefix + 'b/group_norm/gamma']))
            self.gn3.weight.copy_(tf2th(weights[prefix + 'c/group_norm/gamma']))
            self.gn1.bias.copy_(tf2th(weights[prefix + 'a/group_norm/beta']))
            self.gn2.bias.copy_(tf2th(weights[prefix + 'b/group_norm/beta']))
            self.gn3.bias.copy_(tf2th(weights[prefix + 'c/group_norm/beta']))
            if hasattr(self, 'downsample'):
                self.downsample.weight.copy_(tf2th(weights[prefix + 'a/proj/standardized_conv2d/kernel']))
        return self


class ResNetV2(nn.Module):
    BLOCK_UNITS = {
        'r50': [3, 4, 6, 3],
        'r101': [3, 4, 23, 3],
        'r152': [3, 8, 36, 3],
    }

    def __init__(self, block_units, width_factor, head_size=21843, zero_head=False):
        super().__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, 64 * wf, kernel_size=7, stride=2, padding=3, bias=False)),
            ('padp', nn.ConstantPad2d(1, 0)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            # The following is subtly not the same!
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=64 * wf, cout=256 * wf, cmid=64 * wf))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=256 * wf, cout=256 * wf, cmid=64 * wf)) for i in
                 range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=256 * wf, cout=512 * wf, cmid=128 * wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=512 * wf, cout=512 * wf, cmid=128 * wf)) for i in
                 range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=512 * wf, cout=1024 * wf, cmid=256 * wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=1024 * wf, cout=1024 * wf, cmid=256 * wf)) for i in
                 range(2, block_units[2] + 1)],
            ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=1024 * wf, cout=2048 * wf, cmid=512 * wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=2048 * wf, cout=2048 * wf, cmid=512 * wf)) for i in
                 range(2, block_units[3] + 1)],
            ))),
        ]))

        self.zero_head = zero_head
        self.head = nn.Sequential(OrderedDict([
            ('gn', nn.GroupNorm(32, 2048 * wf)),
            ('relu', nn.ReLU(inplace=True)),
            ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
            ('conv', nn.Conv2d(2048 * wf, head_size, kernel_size=1, bias=True)),
        ]))

    def forward(self, x):
        x = self.head(self.body(self.root(x)))
        assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        return x[..., 0, 0]

    def load_from(self, weights, prefix='resnet/'):
        with torch.no_grad():
            self.root.conv.weight.copy_(tf2th(weights[f'{prefix}root_block/standardized_conv2d/kernel']))
            self.head.gn.weight.copy_(tf2th(weights[f'{prefix}group_norm/gamma']))
            self.head.gn.bias.copy_(tf2th(weights[f'{prefix}group_norm/beta']))
            if self.zero_head:
                nn.init.zeros_(self.head.conv.weight)
                nn.init.zeros_(self.head.conv.bias)
            else:
                self.head.conv.weight.copy_(tf2th(weights[f'{prefix}head/conv2d/kernel']))
                self.head.conv.bias.copy_(tf2th(weights[f'{prefix}head/conv2d/bias']))

            for bname, block in self.body.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, prefix=f'{prefix}{bname}/{uname}/')
        return self


def train_model(model, learning_rate, num_epochs, train_loader, test_loader):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Train the model
    loss_arr = []
    total_step = len(train_loader)
    curr_lr = learning_rate
    for epoch in range(num_epochs):
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
        loss_arr.append(loss.item())
        # if (i + 1) % 100 == 0:
        print("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
              .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

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
    torch.save(model.state_dict(), 'resnet.ckpt')
    return model


def dist_loss(t, s, T):
    prob_t = F.softmax(t / T, dim=1)
    log_prob_s = F.log_softmax(s / T, dim=1)
    dist_loss = -(prob_t * log_prob_s).sum(dim=1).mean()
    return dist_loss


def train_KD(train_loader, test_loader, train_dataset, model_teacher, learning_rate, num_epochs):
    Temp_step = [1,2,3,4,5]
    # trying different Temps
    print("the first experiment is trying different T with distillation")
    for t in Temp_step:
        model_1 = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)
        # Training loop
        print("Temperature num is:", t)
        for epoch in range(num_epochs):
            running_loss = 0.0
            running_acc = 0.0
            for data, target in train_loader:
                images = data.to("cuda")
                labels = target.to("cuda")
                optimizer.zero_grad()
                output = model_1(images)
                loss1 = criterion(output, labels)
                loss = loss1 + dist_loss(model_teacher(images).detach(), output, t)
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                running_acc += (predicted == labels).sum().item()
                loss.backward()
                optimizer.step()
            # Print the average loss and accuracy for the epoch
            print(f'Epoch {epoch + 1} Train Loss: {running_loss / len(train_loader)} Train Acc: {(running_acc / len(train_dataset)) * 100}')
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
                    loss = loss1 + dist_loss(teacher_p, outputs, t)
                    val_loss += loss.item()
                    _, pred = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
                val_acc = (correct / total) * 100
                print('Epoch {}:, test loss {}, test acc {}'.format(epoch + 1, val_loss / len(test_loader), val_acc))
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

    # trying different Temps in one run but after some epochs we changed the Temp
    temp_epoch = [[1, 2, 3, 4, 5], [1.5, 2.5, 3.5, 4.5, 5.5]]
    print("The second experiment is put different T after some epoch with distillation")
    model_2 = ResNet(ResidualBlock, [2, 2, 2, 2]).to("cuda")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_2.parameters(), lr=learning_rate)
    # Training loop
    for number in range(2):
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
            if (epoch % 3 == 0):
                t = temp_epoch[number][k]
                k += 1

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


    # trying different alfa in function distillation
    alfa = [0, 0.1, 0.25, 0.5, 0.75, 0.9]
    print("The third experiment is trying different alfa in function distillation")
    t = 3
    for a in alfa:
        model_1 = ResNet(ResidualBlock, [2, 2, 2, 2]).to("cuda")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)
        # Training loop
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
                loss = (a * loss1) + ((1 - a) * dist_loss(model_teacher(images).detach(), output, t))
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                running_acc += (predicted == labels).sum().item()
                loss.backward()
                optimizer.step()
            # Print the average loss and accuracy for the epoch
            print(
                f'Epoch {epoch + 1} Train Loss: {running_loss / len(train_loader)} Train Acc: {(running_acc / len(train_dataset)) * 100}')
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
                    loss = (a * loss1) + ((1 - a) * dist_loss(model_teacher(images).detach(), output, t))
                    val_loss += loss.item()
                    _, pred = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
                val_acc = (correct / total) * 100
                print('Epoch {}:, test loss {}, test acc {}'.format(epoch + 1, val_loss / len(test_loader), val_acc))
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


preprocess_train = tv.transforms.Compose([
    tv.transforms.Resize((160, 160), interpolation=PIL.Image.BILINEAR),
    # It's the default, just being explicit for the reader.
    tv.transforms.RandomCrop((128, 128)),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Get data into [-1, 1]
])

preprocess_eval = tv.transforms.Compose([
    tv.transforms.Resize((128, 128), interpolation=PIL.Image.BILINEAR),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = tv.datasets.CIFAR10(root='./dataCIFAR10', train=True, download=True, transform=preprocess_train)
testset = tv.datasets.CIFAR10(root='./dataCIFAR10', train=False, download=True, transform=preprocess_eval)

train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=128, shuffle=False)

trainset_s = tv.datasets.CIFAR10(root='./cIFAR10', train=True, download=True, transform=tv.transforms.ToTensor())
testset_s = tv.datasets.CIFAR10(root='./cIFAR10', train=False, download=True, transform=tv.transforms.ToTensor())

train_loader_s = torch.utils.data.DataLoader(dataset=trainset_s, batch_size=128, shuffle=True)

test_loader_s = torch.utils.data.DataLoader(dataset=testset_s, batch_size=128, shuffle=False)

weights_cifar10 = get_weights('BiT-M-R50x1-CIFAR10')
model_T = ResNetV2(ResNetV2.BLOCK_UNITS['r50'], width_factor=1, head_size=10)  # NOTE: No new head.
model_T.load_from(weights_cifar10)
model_T.to(device)

model_T = train_model(model_T,0.001,5,train_loader,test_loader)

model_S = ResNet(ResidualBlock, [2, 2, 2, 2]).to(device)
model_S = train_model(model_S,0.001,5,train_loader_s,test_loader_s)

train_KD(train_loader_s,test_loader_s,trainset_s,model_T,0.001,5)
