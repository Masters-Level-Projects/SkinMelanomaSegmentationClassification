# Anurag Banerjee


from torchvision.datasets import ImageFolder
import torchvision.transforms as TF
import torchvision.transforms.functional as TFF
import torch.nn.functional as torchfunc
import torch.nn as nn
import torch
from torch.utils.data.sampler import Sampler
import json
import operator
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as mlines


traindir = 'skin_dis2/train'
valdir = 'skin_dis2/val'
testdir = 'skin_dis2/test'


def load_data():
    tf_lst = [TF.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.2, hue=0.3), TF.RandomHorizontalFlip(p=0.2),
              # TF.RandomAffine(180, translate=None, scale=None, shear=90, resample=False, fillcolor=0), TF.RandomVerticalFlip(p=0.6),
              # TF.RandomAffine(180, translate=None, scale=None, shear=180, resample=False, fillcolor=0), TF.RandomHorizontalFlip(p=0.3),
              TF.RandomRotation(90, resample=False, expand=False, center=None), TF.RandomHorizontalFlip(p=0.2),
              TF.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.2), TF.RandomVerticalFlip(p=0.2)]
    #transforms = TF.Compose([TF.RandomApply(tf_lst), TF.ToTensor(), TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    transforms = TF.Compose([TF.ToTensor(), TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = ImageFolder(traindir, transform=transforms)
    val_data = ImageFolder(valdir, transform=transforms)
    test_data = ImageFolder(testdir, transform=transforms)

    return train_data, val_data, test_data


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            # nn.Conv2d(32, 48, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(48),
            # nn.Conv2d(32, 48, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))

        # self.fc = nn.Linear(111 * 149 * 32, num_classes)
        # self.fc = nn.Linear(55 * 74 * 16, num_classes)
        self.fc = nn.Linear(55 * 74 * 64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = torchfunc.log_softmax(out, dim=1)
        return out


def wts_for_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses

    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = ((N - float(count[i]))/N) * (N/float(count[i]))
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def plot_accuracy_graph(train, val):
    train = [round(elem, 2) for elem in train]
    val = [round(elem, 2) for elem in val]
    fig = plt.figure(1)
    plt.title('Train vs Validation Accuracy Graphs')
    plt.xlabel('epochs', fontsize=7)
    plt.ylabel('accuracy', fontsize=7)

    ymax = max(train + val)
    ymin = min(train + val)
    plt.ylim((ymin, ymax))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    #plt.gca().yaxis().get_offset_text().set_fontsize(5)

    plt.plot(list(range(1, len(train)+1)), train, 'g*-')
    plt.plot(list(range(1, len(train)+1)), val, 'b*-')
    plt.tick_params(axis='both', labelsize=6)
    trainln = mlines.Line2D([], [], color='green', marker='*', markersize=6)
    valln = mlines.Line2D([], [], color='blue', marker='*', markersize=6)
    fig.legend(handles=(trainln, valln), labels=('Train', 'Val'), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.show()


def plot_loss_graph(train, val):
    train = [round(elem, 2) for elem in train]
    val = [round(elem, 2) for elem in val]
    fig = plt.figure(2)
    plt.title('Train vs Validation Loss Graphs')
    plt.xlabel('epochs', fontsize=7)
    plt.ylabel('loss', fontsize=7)

    ymax = max(train + val)
    ymin = min(train + val)
    plt.ylim((ymin, ymax))
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    #plt.gca().yaxis().get_offset_text().set_fontsize(5)

    plt.plot(list(range(1, len(train) + 1)), train, 'g*-')
    plt.plot(list(range(1, len(train) + 1)), val, 'b*-')
    plt.tick_params(axis='both', labelsize=6)
    trainln = mlines.Line2D([], [], color='green', marker='*', markersize=6)
    valln = mlines.Line2D([], [], color='blue', marker='*', markersize=6)
    fig.legend(handles=(trainln, valln), labels=('Train', 'Val'), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.show()


def main():
    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Hyper parameters
    num_epochs = 40
    num_classes = 7
    batch_size = 16
    learning_rate = 0.0005
    wt_dcy = 1e-1

    # traind, vald, testd = load_data()       # access the actual files on my disk
    traind, vald, testd = load_data()  # access the actual files on my disk

    # weighted sampling
    weights = wts_for_classes(traind.imgs, len(traind.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=traind, sampler=sampler, batch_size=batch_size, shuffle=False,
                                               pin_memory=True, num_workers=1, drop_last=True)
    # shuffle is mutually exclusive with sampler

    val_loader = torch.utils.data.DataLoader(dataset=vald, sampler=None, batch_size=batch_size, shuffle=False,
                                             pin_memory=True, num_workers=1)
    tst_loader = torch.utils.data.DataLoader(dataset=testd, sampler=None, batch_size=batch_size, shuffle=False,
                                             pin_memory=True, num_workers=1)

    model = ConvNet(num_classes).to(device)

    # Loss and optimizer
    perclass_count = []
    ttl_cnt = 0
    perclass_prob = []
    class_idx_map = train_loader.dataset.class_to_idx
    with open('datadict', 'r') as fin:
        datadict = json.load(fin)
        for key, _ in sorted(class_idx_map.items(), key=operator.itemgetter(1)):
            perclass_count.append(datadict[key])
            ttl_cnt += datadict[key]
    for i in range(len(perclass_count)):
        perclass_prob.append(1 - (perclass_count[i] / ttl_cnt)*0.4)

    weights = torch.FloatTensor(np.asarray(perclass_prob)).cuda()

    criterion = nn.CrossEntropyLoss(weight=weights)
    # criterion = nn.MSELoss()
    # criterion = nn.NLLLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wt_dcy)

    torch.cuda.empty_cache()
    # Train the model
    print("\nBegin training the model...")
    total_step = len(train_loader)

    # variables for storing loss and accuracy per epoch
    e_train_loss = []
    e_train_acc = []
    e_val_loss = []
    e_val_acc = []

    for epoch in range(num_epochs):
        e_tr_l = 0
        e_vl_l = 0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            trainloss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            trainloss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            e_tr_l += trainloss.item()

            if (i + 1) % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'
                      .format(epoch + 1, num_epochs, i + 1, total_step, trainloss.item()))
        e_train_loss.append(e_tr_l/len(train_loader))
        e_train_acc.append(correct/total)
        # Validate the model
        model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            count = 0
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                count += len(images)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                valloss = criterion(outputs, labels)

                e_vl_l += valloss.item()

            print('Validation Accuracy of the model on {} validation images: {:.3f} %'.format(count, 100 * correct / total))
        model.train()
        e_val_loss.append(e_vl_l / len(val_loader))
        e_val_acc.append(correct/total)

    # Save the model checkpoint
    checkpoint = {'model': ConvNet(),
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'criterion': criterion}
    torch.save(checkpoint, 'model_cnn_pytorch3.ckpt')

    print("\nTraining complete! Now finally testing learned model")

    plot_accuracy_graph(e_train_acc, e_val_acc)
    plot_loss_graph(e_train_loss, e_val_loss)
    from tester import doTest
    doTest(tst_loader)


if __name__ == '__main__':
    main()
