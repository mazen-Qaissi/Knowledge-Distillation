import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

train_dataset = torchvision.datasets.CIFAR100(root='./dataCIFAR100', train=True, download=True,
                                              transform=train_transform)

test_dataset = torchvision.datasets.CIFAR100(root='./dataCIFAR100', train=False, download=True,
                                             transform=test_transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, cfg, name="student", batch_norm=False, num_classes=1000):
        super(VGG, self).__init__()
        self.name = name
        self.block0 = self._make_layers(cfg[0], batch_norm, 3)
        self.block1 = self._make_layers(cfg[1], batch_norm, cfg[0][-1])
        self.block2 = self._make_layers(cfg[2], batch_norm, cfg[1][-1])
        self.block3 = self._make_layers(cfg[3], batch_norm, cfg[2][-1])
        self.block4 = self._make_layers(cfg[4], batch_norm, cfg[3][-1])

        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.block0)
        feat_m.append(self.pool0)
        feat_m.append(self.block1)
        feat_m.append(self.pool1)
        feat_m.append(self.block2)
        feat_m.append(self.pool2)
        feat_m.append(self.block3)
        feat_m.append(self.pool3)
        feat_m.append(self.block4)
        feat_m.append(self.pool4)
        return feat_m

    def get_bn_before_relu(self):
        bn1 = self.block1[-1]
        bn2 = self.block2[-1]
        bn3 = self.block3[-1]
        bn4 = self.block4[-1]
        return [bn1, bn2, bn3, bn4]

    def forward(self, x, y_t=None, is_feat=False, preact=False, detach_body=False):
        h = x.shape[2]
        x = F.relu(self.block0(x))
        f0 = x
        x = self.pool0(x)
        x = self.block1(x)
        f1_pre = x
        x = F.relu(x)
        f1 = x

        x = self.pool1(x)
        x = self.block2(x)
        f2_pre = x
        x = F.relu(x)
        f2 = x

        x = self.pool2(x)
        x = self.block3(x)
        f3_pre = x
        x = F.relu(x)
        f3 = x

        if h == 64:
            x = self.pool3(x)

        x = self.block4(x)
        f4_pre = x
        x = F.relu(x)
        f4 = x

        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        f5 = x

        x = self.classifier(x)

        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], x
            else:
                return [f0, f1, f2, f3, f4, f5], x
        else:
            return x

    @staticmethod
    def _make_layers(cfg, batch_norm=False, in_channels=3):
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        layers = layers[:-1]
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


cfg = {
    'A': [[64], [128], [256, 256], [512, 512], [512, 512]],
    'B': [[64, 64], [128, 128], [256, 256], [512, 512], [512, 512]],
    'D': [[64, 64], [128, 128], [256, 256, 256], [512, 512, 512], [512, 512, 512]],
    'E': [[64, 64], [128, 128], [256, 256, 256, 256], [512, 512, 512, 512], [512, 512, 512, 512]],
    'S': [[64], [128], [256], [512], [512]],
}


def vgg8(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], **kwargs)
    return model


def vgg8_bn(**kwargs):
    """VGG 8-layer model (configuration "S")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['S'], batch_norm=True, **kwargs)
    return model


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['A'], **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(cfg['A'], batch_norm=True, **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['B'], **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(cfg['B'], batch_norm=True, **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['D'], **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(cfg['D'], batch_norm=True, **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(cfg['E'], **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(cfg['E'], batch_norm=True, **kwargs)
    return model


def train_model(model, learning_rate, num_epochs, train_loader, test_loader):
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
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
    # Save the model checkpoint
    torch.save(model.state_dict(), 'resnet.ckpt')
    return model


def dist_loss(t, s, T):
    prob_t = F.softmax(t / T, dim=1)
    log_prob_s = F.log_softmax(s / T, dim=1)
    dist_loss = -(prob_t * log_prob_s).sum(dim=1).mean()
    return dist_loss


def train_KD(train_loader, test_loader, train_dataset,test_dataset, model_teacher, learning_rate, num_epochs):
    Temp_step = [1, 2, 3, 4, 5]
    # # trying different Temps
    print("the first experiment is trying different T with distillation")
    for t in Temp_step:
        criterion_T = nn.CrossEntropyLoss()
        optimizer_T = torch.optim.SGD(model_teacher.parameters(), lr=learning_rate, momentum=0.9)
        model = vgg8_bn(num_classes=100).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        # Training loop
        for epoch in range(num_epochs):
            # teacher
            running_loss_T = 0.0
            running_acc_T = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to("cuda")
                labels = labels.to("cuda")
                optimizer_T.zero_grad()
                outputs_T = model_teacher(inputs)
                loss_T = criterion_T(outputs_T, labels)
                running_loss_T += loss_T.item()
                _, predicted = torch.max(outputs_T.data, 1)
                running_acc_T += (predicted == labels).sum().item()
                loss_T.backward()
                optimizer_T.step()
            # Print the average loss and accuracy for the epoch
            print(
                f'Teacher Epoch {epoch + 1} Loss: {running_loss_T / len(train_loader)} Acc: {(running_acc_T / len(train_dataset)) * 100}')
            # Testing loop
            with torch.no_grad():
                test_loss_T = 0.0
                test_acc_T = 0.0
                for data in test_loader:
                    inputs, labels = data
                    inputs = inputs.to("cuda")
                    labels = labels.to("cuda")
                    outputs = model_teacher(inputs)
                    test_loss_T += criterion_T(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_acc_T += (predicted == labels).sum().item()

            # Print the average loss and accuracy for the test set
            print(f'Teacher Test Loss: {test_loss_T / len(test_loader)} Test Acc: {(test_acc_T / len(test_dataset)) * 100}')
            # distillation
            running_loss = 0.0
            running_acc = 0.0
            for data, target in train_loader:
                images = data.to("cuda")
                labels = target.to("cuda")
                optimizer.zero_grad()
                output = model(images)
                loss1 = criterion(output, labels)
                loss = loss1 + dist_loss(model_teacher(images), output, t)
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                running_acc += (predicted == labels).sum().item()
                loss.backward()
                optimizer.step()
            # Print the average loss and accuracy for the epoch
            print(
                f'Distillation Epoch {epoch + 1} Train Loss: {running_loss / len(train_loader)} Train Acc: {(running_acc / len(train_dataset)) * 100}')
            # Testing loop
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    loss1 = F.cross_entropy(outputs, labels)
                    teacher_p = model_teacher(inputs)
                    loss = loss1 + dist_loss(teacher_p, outputs, t)
                    val_loss += loss.item()
                    _, pred = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
                val_acc = (correct / total) * 100
                print('Distillation Epoch {}:, test loss {}, test acc {}'.format(epoch + 1, val_loss / len(test_loader),val_acc))
        model_teacher.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model_teacher(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the model of teacher on the test images: {} %'.format(100 * correct / total))
            for images, labels in train_loader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model_teacher(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of the model of teacher on the train images: {} %'.format(100 * correct / total))
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the model of distillation on the test images: {} %'.format(100 * correct / total))
            for images, labels in train_loader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of the model of distillation on the train images: {} %'.format(100 * correct / total))

    temp_epoch = [[1, 2, 3, 4, 5],[1.5, 2.5, 3.5, 4.5, 5.5]]
    print("The second experiment is put different T after some epoch with distillation")
    k = 0
    for set_t in temp_epoch:
        criterion_T = nn.CrossEntropyLoss()
        optimizer_T = torch.optim.SGD(model_teacher.parameters(), lr=learning_rate, momentum=0.9)
        model = vgg8_bn(num_classes=100).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
        print("Temperature set is:", set_t)
        # Training loop
        for epoch in range(num_epochs):
            if epoch % 3 == 0:
                t = set_t[k]
                k = + 1
            # teacher
            running_loss_T = 0.0
            running_acc_T = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to("cuda")
                labels = labels.to("cuda")
                optimizer_T.zero_grad()
                outputs_T = model_teacher(inputs)
                loss_T = criterion_T(outputs_T, labels)
                running_loss_T += loss_T.item()
                _, predicted = torch.max(outputs_T.data, 1)
                running_acc_T += (predicted == labels).sum().item()
                loss_T.backward()
                optimizer_T.step()
            # Print the average loss and accuracy for the epoch
            print(
                f'Teacher Epoch {epoch + 1} Loss: {running_loss_T / len(train_loader)} Acc: {(running_acc_T / len(train_dataset)) * 100}')
            # Testing loop
            with torch.no_grad():
                test_loss_T = 0.0
                test_acc_T = 0.0
                for data in test_loader:
                    inputs, labels = data
                    inputs = inputs.to("cuda")
                    labels = labels.to("cuda")
                    outputs = model_teacher(inputs)
                    test_loss_T += criterion_T(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_acc_T += (predicted == labels).sum().item()

            # Print the average loss and accuracy for the test set
            print(
                f'Teacher Test Loss: {test_loss_T / len(test_loader)} Test Acc: {(test_acc_T / len(test_dataset)) * 100}')
            # distillation
            running_loss = 0.0
            running_acc = 0.0
            for data, target in train_loader:
                images = data.to("cuda")
                labels = target.to("cuda")
                optimizer.zero_grad()
                output = model(images)
                loss1 = criterion(output, labels)
                loss = loss1 + dist_loss(model_teacher(images), output, t)
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                running_acc += (predicted == labels).sum().item()
                loss.backward()
                optimizer.step()
            # Print the average loss and accuracy for the epoch
            print(
                f'Distillation Epoch {epoch + 1} Train Loss: {running_loss / len(train_loader)} Train Acc: {(running_acc / len(train_dataset)) * 100}')
            # Testing loop
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    loss1 = F.cross_entropy(outputs, labels)
                    teacher_p = model_teacher(inputs)
                    loss = loss1 + dist_loss(teacher_p, outputs, t)
                    val_loss += loss.item()
                    _, pred = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
                val_acc = (correct / total) * 100
                print('Distillation Epoch {}:, test loss {}, test acc {}'.format(epoch + 1, val_loss / len(test_loader),
                                                                                 val_acc))
        # //-----------------------------------------------------------------------------
        model_teacher.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model_teacher(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the model of teacher on the test images: {} %'.format(100 * correct / total))
            for images, labels in train_loader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model_teacher(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of the model of teacher on the train images: {} %'.format(100 * correct / total))
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the model of distillation on the test images: {} %'.format(100 * correct / total))
            for images, labels in train_loader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of the model of distillation on the train images: {} %'.format(100 * correct / total))

    alpha = [0, 0.1, 0.25, 0.5, 0.75, 0.9]
    t = 3
    # # trying different Temps
    print("The third experiment is trying different alfa in function distillation")
    for a in alpha:
        print("Alfa is:", a)
        criterion_T = nn.CrossEntropyLoss()
        optimizer_T = torch.optim.SGD(model_teacher.parameters(), lr=learning_rate, momentum=0.9)
        model = vgg8_bn(num_classes=100).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

        # Training loop
        for epoch in range(num_epochs):
            # teacher
            running_loss_T = 0.0
            running_acc_T = 0.0
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs = inputs.to("cuda")
                labels = labels.to("cuda")
                optimizer_T.zero_grad()
                outputs_T = model_teacher(inputs)
                loss_T = criterion_T(outputs_T, labels)
                running_loss_T += loss_T.item()
                _, predicted = torch.max(outputs_T.data, 1)
                running_acc_T += (predicted == labels).sum().item()
                loss_T.backward()
                optimizer_T.step()
            # Print the average loss and accuracy for the epoch
            print(
                f'Teacher Epoch {epoch + 1} Loss: {running_loss_T / len(train_loader)} Acc: {(running_acc_T / len(train_dataset)) * 100}')
            # Testing loop
            with torch.no_grad():
                test_loss_T = 0.0
                test_acc_T = 0.0
                for data in test_loader:
                    inputs, labels = data
                    inputs = inputs.to("cuda")
                    labels = labels.to("cuda")
                    outputs = model_teacher(inputs)
                    test_loss_T += criterion_T(outputs, labels).item()
                    _, predicted = torch.max(outputs.data, 1)
                    test_acc_T += (predicted == labels).sum().item()

            # Print the average loss and accuracy for the test set
            print(
                f'Teacher Test Loss: {test_loss_T / len(test_loader)} Test Acc: {(test_acc_T / len(test_dataset)) * 100}')
            # distillation
            running_loss = 0.0
            running_acc = 0.0
            for data, target in train_loader:
                images = data.to("cuda")
                labels = target.to("cuda")
                optimizer.zero_grad()
                output = model(images)
                loss1 = criterion(output, labels)
                loss = (a * loss1) + ((1 - a) * dist_loss(model_teacher(images).detach(), output, t))
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                running_acc += (predicted == labels).sum().item()
                loss.backward()
                optimizer.step()
            # Print the average loss and accuracy for the epoch
            print(
                f'Distillation Epoch {epoch + 1} Train Loss: {running_loss / len(train_loader)} Train Acc: {(running_acc / len(train_dataset)) * 100}')
            # Testing loop
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.cuda(), labels.cuda()
                    outputs = model(inputs)
                    loss1 = F.cross_entropy(outputs, labels)
                    teacher_p = model_teacher(inputs)
                    loss = loss1 + dist_loss(teacher_p, outputs, t)
                    val_loss += loss.item()
                    _, pred = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (pred == labels).sum().item()
                val_acc = (correct / total) * 100
                print('Distillation Epoch {}:, test loss {}, test acc {}'.format(epoch + 1, val_loss / len(test_loader),val_acc))
        model_teacher.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model_teacher(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the model of teacher on the test images: {} %'.format(100 * correct / total))
            for images, labels in train_loader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model_teacher(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print(
                'Accuracy of the model of teacher on the train images: {} %'.format(100 * correct / total))
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            print('Accuracy of the model of distillation on the test images: {} %'.format(
                100 * correct / total))
            for images, labels in train_loader:
                images = images.to("cuda")
                labels = labels.to("cuda")
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            print('Accuracy of the model of distillation on the train images: {} %'.format(
                100 * correct / total))


if __name__ == '__main__':
    import torch

    model_vgg8 = vgg8_bn(num_classes=100).to(device)
    model_vgg13 = vgg13_bn(num_classes=100).to(device)
    log_S = train_model(model_vgg8, 0.05, 40, train_loader, test_loader)
    log_T = train_model(model_vgg13, 0.05, 40, train_loader, test_loader)
    train_KD(train_loader,test_loader,train_dataset,test_dataset,model_vgg13,0.05,40)
