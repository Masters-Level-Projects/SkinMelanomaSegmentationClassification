# Anurag Banerjee

# import torchvision.transforms as tf
import pandas as pd
import os
import shutil
import random


label_path = 't3label/ISIC2018_Task3_Training_GroundTruth.csv'
data_path = 't3data/'
EXT = '.jpg'
ImgSet = './skin_dis2/'
train_perc = 0.80
val_perc = 0.10
test_perc = 0.10


def read_csv_file():
    labelInfo = pd.read_csv(label_path)     # read the CSV file and return the data
    return labelInfo


def preplabeldict(lblInfo):
    labels = lblInfo.columns.tolist()[1:]   # prior knowledge, columns are IMAGE LBL1 LBL2 ...

    label_dict = {}
    for lbl in labels:
        label_dict[lbl] = []

    for idx, row in lblInfo.iterrows():
        for lbl in labels:
            if row[lbl] == 1.0:
                label_dict[lbl].append(row.iloc[0])
    return label_dict


def prepfolder(lbl_dict, lbl_Info):
    if not os.path.exists(ImgSet):
        os.mkdir(ImgSet)  # creating the image folder to be consumed by PyTorch ImageLoader

    for key in lbl_dict.keys():
        if not os.path.exists(ImgSet+'train/'+key):
            os.makedirs(ImgSet+'train/'+key)
        if not os.path.exists(ImgSet + 'val/'+key):
            os.makedirs(ImgSet + 'val/' + key)
        if not os.path.exists(ImgSet + 'test/'+key):
            os.makedirs(ImgSet + 'test/' + key)

    for key in lbl_dict.keys():
        print('\nCopying files for label: '+key)
        lst_sz = len(lbl_dict[key])     # get the size of file list under this label
        train_sz = round(lst_sz * train_perc)   # get size for train partition
        val_sz = round(lst_sz * val_perc)
        test_sz = round(lst_sz * test_perc)     # get size for test partition

        filelst = lbl_dict[key]
        random.shuffle(filelst)   # shuffle the current file list
        trainlst = filelst[:train_sz]           # get train_sz number of elements
        vallst = filelst[train_sz: train_sz + val_sz]
        testlst = filelst[train_sz + val_sz: train_sz + val_sz + test_sz]


        # now copy the files from source to destination
        # first copy the test files
        for file in testlst:
            src = data_path + file + EXT
            dest = ImgSet + 'test/' + key
            shutil.copy(src, dest)

        # next copy the train files
        for file in trainlst:
            src = data_path + file + EXT
            dest = ImgSet + 'train/' + key
            shutil.copy(src, dest)

        # next copy the val files
        for file in vallst:
            src = data_path + file + EXT
            dest = ImgSet + 'val/' + key
            shutil.copy(src, dest)


def main():
    # transform = tf.Compose([tf.ToTensor(), tf.normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    labelInfo = read_csv_file()          # read label info from a csv file
    lbl_dict = preplabeldict(labelInfo)
    prepfolder(lbl_dict, labelInfo)
    # sum = 0
    # for key in lbl_dict.keys():
    #     print(len(lbl_dict[key]))
    #     sum = sum + len(lbl_dict[key])
    #
    # print('Sum = '+str(sum))


if __name__ == '__main__':
    main()
