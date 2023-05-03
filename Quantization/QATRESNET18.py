import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import copy
from resnet import resnet18

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prepare_dataloader(train_batch_size=256, eval_batch_size=256):
    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # ])

    train_transform = transforms.ToTensor()

    # test_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #     transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # ])
    #
    test_transform = transforms.ToTensor()

    train_set = torchvision.datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)

    test_set = torchvision.datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_batch_size)

    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=eval_batch_size)

    return train_loader, test_loader, train_set, test_set


def evaluate_model(model, test_loader, device, criterion=None):
    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy


def train_model(model, train_loader, test_loader, device, learning_rate=1e-1, num_epochs=20):
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Evaluation
    model.eval()
    eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device, criterion=criterion)
    print("Epoch: {:02d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(-1, eval_loss, eval_accuracy))

    for epoch in range(num_epochs):

        # Training
        model.train()

        running_loss = 0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        train_loss = running_loss / len(train_loader.dataset)
        train_accuracy = running_corrects / len(train_loader.dataset)

        # Evaluation
        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_loader, device=device,
                                                  criterion=criterion)

        print("Epoch: {:03d} Train Loss: {:.3f} Train Acc: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(epoch,
                                                                                                             train_loss,
                                                                                                             train_accuracy,
                                                                                                             eval_loss,
                                                                                                             eval_accuracy))

    return model


def print_size_of_model(model):
    """ Prints the real size of the model """
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p") / 1e6)
    os.remove('temp.p')



def create_model():
    model = resnet18(pretrained=False)
    return model


class QuantizedResNet18(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedResNet18, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = model_fp32

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x


def main():
    model = create_model()
    print_size_of_model(model)
    train_loader, test_loader, train_set, test_set = prepare_dataloader(train_batch_size=256, eval_batch_size=256)
    model = train_model(model,train_loader,test_loader,device,0.002,25)
    loss_test_R, acc_test_R = evaluate_model(model, test_loader, device, nn.CrossEntropyLoss())
    print("Resnet18 after 25 epoch, Loss: {:.3f}, Acc: {:.3f}".format(loss_test_R,(acc_test_R*100)))
    fused_model = copy.deepcopy(model)
    model.train()
    fused_model.eval().to("cpu")

    fused_model = torch.ao.quantization.fuse_modules(fused_model, [["conv1", "bn1", "relu"]], inplace=True)
    for module_name, module in fused_model.named_children():
        if "layer" in module_name:
            for basic_block_name, basic_block in module.named_children():
                torch.ao.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu1"], ["conv2", "bn2"]],
                                                inplace=True)
                for sub_block_name, sub_block in basic_block.named_children():
                    if sub_block_name == "downsample":
                        torch.ao.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)

    quantized_model = QuantizedResNet18(model_fp32=fused_model)
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    quantized_model.qconfig = quantization_config
    torch.quantization.prepare_qat(quantized_model, inplace=True)
    print("Training QAT Model...")
    quantized_model.eval()
    train_model(model=quantized_model, train_loader=train_loader, test_loader=test_loader,device="cuda",learning_rate=0.002, num_epochs=25)
    quantized_model.to("cpu")
    quantized_model = torch.quantization.convert(quantized_model, inplace=True)
    print_size_of_model(quantized_model)
    loss_test_q, acc_test_q = evaluate_model(quantized_model, test_loader, "cpu", nn.CrossEntropyLoss())
    print("Quantization Resnet18 after 25 epoch, Loss: {:.3f}, Acc: {:.3f}".format(loss_test_q, (acc_test_q*100)))


if __name__ == "__main__":
    main()