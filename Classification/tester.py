from torchvision.datasets import ImageFolder
import torchvision.transforms as TF
import torch
from torch.utils.data.sampler import Sampler
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pandas
import seaborn


traindir = 'skin_dis2/train'
valdir = 'skin_dis2/val'
testdir = 'skin_dis2/test'


def load_data():
    tf_lst = [TF.ColorJitter(brightness=0.5, contrast=0.1, saturation=0.2, hue=0.3), TF.RandomHorizontalFlip(p=0.1),
              # TF.RandomAffine(180, translate=None, scale=None, shear=90, resample=False, fillcolor=0), TF.RandomVerticalFlip(p=0.6),
              # TF.RandomAffine(180, translate=None, scale=None, shear=180, resample=False, fillcolor=0), TF.RandomHorizontalFlip(p=0.3),
              TF.RandomRotation(90, resample=False, expand=False, center=None), TF.RandomHorizontalFlip(p=0.5),
              TF.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.2), TF.RandomVerticalFlip(p=0.2)]
    transforms = TF.Compose([TF.RandomApply(tf_lst), TF.ToTensor(), TF.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = ImageFolder(traindir, transform=transforms)
    val_data = ImageFolder(valdir, transform=transforms)
    test_data = ImageFolder(testdir, transform=transforms)

    return train_data, val_data, test_data


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


def plot_confusion(cm,allClasses):
    dataframe_cm = pandas.DataFrame(cm,allClasses,allClasses)
    centerVal = np.max(cm)/2
    print('Plotting Confusion Matrix')
    fig = plt.figure()
    plt.subplot(1, 1, 1)
    seaborn.heatmap(dataframe_cm,fmt='g',annot=True,annot_kws={"size": 8},center=centerVal,vmin=0,vmax=100)
    plt.ylabel('Target Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix of Predicted Classes')
    plt.tight_layout()
    plt.show()


def doTest(tst_loader):
    model = load_checkpoint('model_cnn_pytorch3.ckpt')

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if tst_loader is None:
        traind, vald, testd = load_data()  # access the actual files on my disk
        batch_size = 16
        tst_loader = torch.utils.data.DataLoader(dataset=testd, sampler=None, batch_size=batch_size, shuffle=False,
                                                 pin_memory=True, num_workers=1)

    model.to(device)

    print('Now evaluating the model...')

    Y_act = []
    Y_pred = []
    with torch.no_grad():
        correct = 0
        total = 0
        count = 0
        for images, labels in tst_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            count += len(images)

            pred_label = torch.argmax(outputs.data, 1)
            truth = labels.cpu().numpy()
            obs = pred_label.cpu().numpy()
            Y_act += truth.tolist()
            Y_pred += obs.tolist()

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the {} test images: {} %'.format(count, 100 * correct / total))
    assert (len(Y_pred) == len(Y_act))
    print('We will now have a look at the confusion matrix for the test set...')
    cm = confusion_matrix(np.array(Y_act), np.array(Y_pred))
    all_classes = list(tst_loader.dataset.class_to_idx.keys())

    row_sum = np.sum(cm, axis=1)
    row_sum = np.reshape(row_sum, (len(all_classes), 1))
    cm = (cm / row_sum) * 100

    plot_confusion(cm, all_classes)


if __name__ == '__main__':

    doTest(None)
