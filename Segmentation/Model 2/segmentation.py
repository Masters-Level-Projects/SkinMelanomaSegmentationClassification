import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy
import time
import random
import matplotlib.pyplot as plt
import os
from torchvision.models.vgg import VGG
from torchvision import transforms
import PIL

# Seeding the random states with a particular value. Useful for reproducibility of the result.
def random_seed(seed_value, use_cuda):
    numpy.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

# Format the Calculated Time in Hours:Minutes:Seconds Format.
def format_time(time_amount):
    hrs = int(time_amount/3600)
    time_amount = time_amount % 3600
    mins = int(time_amount/60)
    secs = round((time_amount % 60),0)
    time_taken = str(hrs) + ' hrs ' + str(mins) + ' mins ' + str(secs) + ' secs.'
    return(time_taken)
    
# Get FileNames
def getFileNames(path,subpath):
    total_path = path + subpath
    namelist = os.listdir(total_path)
    name_number = []
    for i in range(len(namelist)): name_number.append(namelist[i][5:12])
    name_number = numpy.array(name_number,dtype=numpy.int32)
    #name_number = name_number[0:100]
    return(name_number,name_number)

class SpatialDatasetNames(Dataset):
    
    def __init__(self,imgName,flags,X,Y):
        self.len = imgName.shape[0]
        self.imgName_data = torch.from_numpy(imgName)
        self.x_data = torch.from_numpy(X)
        self.y_data = torch.from_numpy(Y)
        self.flags = torch.from_numpy(flags)
    
    #def __getitem__(self, index): return self.imgName_data[index], self.x_data[index], self.y_data[index]
        
    def __getitem__(self, index):
    
        temp_x = self.x_data[index]
        temp_y = self.y_data[index]
        
        transformationPIL = transforms.ToPILImage()
        transformationTensor = transforms.ToTensor()
    
        if self.flags[index] == 1:
            def perform_transformation_operation(temp_x,temp_y):
                threshold = 0.5
                threshold_angle = 0.2
                if threshold_angle <= random.random() == True:
                    angle = random.randint(-45, 45)
                    temp_x = F.rotate(temp_x, angle)
                    temp_y = F.rotate(temp_y, angle)
                if threshold <= random.random() == True:
                    temp_x = transforms.functional.hflip(temp_x)
                    temp_y = transforms.functional.hflip(temp_y)
                if threshold <= random.random() == True:
                    temp_x = transforms.functional.hflip(temp_x)
                    temp_y = transforms.functional.hflip(temp_y)
                return(temp_x,temp_y)
        
            #print(temp_x.type())
            #print(temp_y.type())
            #print(temp_x.size())
            
            temp_x = transformationPIL(temp_x)
            temp_y = transformationPIL(temp_y)

            threshold = 0.2
        
            if threshold <= random.random() == True:
                brightness_factor = 0.7 + (random.random()*0.6)
                temp_x = transforms.functional.adjust_brightness(temp_x, brightness_factor=brightness_factor)
        
            if threshold <= random.random() == True:
                contrast_factor = 0.4 + (random.random()*1.2)
                temp_x = transforms.functional.adjust_contrast(temp_x, contrast_factor=contrast_factor)
        
            if threshold <= random.random() == True:
                gamma = 0.6 + (random.random()*0.8)
                temp_x = transforms.functional.adjust_gamma(temp_x, gamma=gamma, gain=1)
        
            #if threshold <= random.random() == True:
                #hue_factor = 0.7 + (random.random()*0.6)
                #temp_x = transforms.functional.adjust_hue(temp_x, hue_factor=hue_factor)
        
            if threshold <= random.random() == True:
                saturation_factor = 0.8 + (random.random()*0.4)
                temp_x = transforms.functional.adjust_saturation(temp_x, saturation_factor=saturation_factor)
         
            temp_x,temp_y = perform_transformation_operation(temp_x,temp_y)
        
            temp_x = transformationTensor(temp_x)
            temp_y = transformationTensor(temp_y)
        
            #temp_x = temp_x/255
            #temp_y = temp_y/255
        
            #print(temp_x.type())
            #print(temp_y.type())
            #print(temp_y.size())
        
        else:
            temp_x = transformationPIL(temp_x)
            temp_y = transformationPIL(temp_y)
            temp_x = transforms.functional.adjust_contrast(temp_x, contrast_factor=1.3)
            #temp_x = transforms.functional.adjust_hue(temp_x, hue_factor=0.75)
            temp_x = transformationTensor(temp_x)
            temp_y = transformationTensor(temp_y)
        
        return self.imgName_data[index], temp_x, temp_y
    
    def __len__(self): return self.len
  
  
    
def double_conv3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True)
    )
    
def double_conv7(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 11, padding=5),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 11, padding=5),
        nn.ReLU(inplace=True)
    )
    
def double_conv5(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 11, padding=5),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 11, padding=5),
        nn.ReLU(inplace=True)
    )
    
def single_conv1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, padding=0),
        nn.ReLU(inplace=True)
    )   

class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv3(3, 32)
        self.dconv_down2 = double_conv3(32, 64)
        self.dconv_down3 = double_conv3(64, 128)
        self.dconv_down4 = double_conv3(128, 256)
        self.dconv_down5 = double_conv3(256, 512)
        
        self.dconv_up4 = double_conv3(256 + 512, 256)
        self.dconv_up3 = double_conv3(128 + 256, 128)
        self.dconv_up2 = double_conv3(128 + 64, 64)
        self.dconv_up1 = double_conv3(64 + 32, 32)
        
        self.scn_layer1_down = single_conv1(3, 32)
        self.scn_layer2_down = single_conv1(3, 64)
        self.scn_layer3_down = single_conv1(3, 128)
        self.scn_layer4_down = single_conv1(3, 256)
        self.scn_layer5_down = single_conv1(3, 512)
        
        self.scn_layer4_up = single_conv1(512, 256)
        self.scn_layer3_up = single_conv1(512, 128)
        self.scn_layer2_up = single_conv1(512, 64)
        self.scn_layer1_up = single_conv1(512, 32)

        self.conv7_post = double_conv7(32, 16)
        self.conv5_post = double_conv5(32, 16)
        self.conv3_post = double_conv3(32, 32)
        
        self.scn_layer_skip = single_conv1(32, 64)
        self.scn_layer_imgCarry = single_conv1(3, 64)
        
        #self.conv1_enc = single_conv1(16 + 16 + 32, 32)
        
        self.maxpool = nn.MaxPool2d(2)
        self.avgpool = nn.AvgPool2d(2)
        self.dropout = nn.Dropout(p=0.2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
    
        x_duplicate = x
        #print(x_duplicate.size())
        
        scn = self.scn_layer1_down(x_duplicate)
        scn = self.avgpool(scn)
        
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        x = self.dropout(x)
        x = x + scn

        scn = self.scn_layer2_down(x_duplicate)
        scn = self.avgpool(self.avgpool(scn))
        
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        x = x + scn
        
        scn = self.scn_layer3_down(x_duplicate)
        scn = self.avgpool(self.avgpool(scn))
        scn = self.avgpool(scn)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        x = x + scn
        
        scn = self.scn_layer4_down(x_duplicate)
        scn = self.avgpool(self.avgpool(scn))
        scn = self.avgpool(self.avgpool(scn))
        
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = x + scn
        
        scn = self.scn_layer5_down(x_duplicate)
        scn = self.avgpool(self.avgpool(scn))
        scn = self.avgpool(self.avgpool(scn))
        
        x = self.dconv_down5(x)
        x = self.dropout(x)
        x = x + scn
        
        x_triplicate = x
        #print(x_triplicate.size())
        
        scn = self.scn_layer4_up(x_triplicate)
        scn = self.upsample(self.upsample(scn))
        
        x = self.upsample(x) 
        x = torch.cat([x, conv4], dim=1)
        
        
        x = self.dconv_up4(x)
        x = self.upsample(x)
        #print(scn.size())
        #print(x.size())
        x = x + scn    
        x = torch.cat([x, conv3], dim=1)
        
        scn = self.scn_layer3_up(x_triplicate)
        scn = self.upsample(self.upsample(scn))
        scn = self.upsample(scn)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = x + scn  
        x = torch.cat([x, conv2], dim=1)
        
        scn = self.scn_layer2_up(x_triplicate)
        scn = self.upsample(self.upsample(scn))
        scn = self.upsample(self.upsample(scn))

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = x + scn 
        x = torch.cat([x, conv1], dim=1)
        
        scn = self.scn_layer1_up(x_triplicate)
        scn = self.upsample(self.upsample(scn))
        scn = self.upsample(self.upsample(scn))
        
        x = self.dconv_up1(x)
        x = self.dropout(x)
        #print(scn.size())
        #print(x.size())
        x = x + scn
        
        x_uOutput = self.scn_layer_skip(x)
        
        x_7 = self.conv7_post(x)
        x_5 = self.conv5_post(x)
        x_3 = self.conv3_post(x)
        
        x = torch.cat([x_7, x_5, x_3], dim=1)
        
        x_carry = self.scn_layer_imgCarry(x_duplicate)
        x_carry = self.dropout(x_carry)
        
        x = x + x_uOutput + x_carry
        #x = self.conv1_enc(x)
        #x = fcn + x
        
        #x = self.fcn_layer5(x)
        
        out = self.conv_last(x)
        
        return out

def get_Images(fileNames,c):
    if c == 1: X = numpy.zeros((len(fileNames),height,width,3),dtype=numpy.uint8)
    else: X = numpy.zeros((len(fileNames),height,width),dtype=numpy.uint8)
    for i,item in enumerate(fileNames):
        if c == 1:
            fileName = path + '/Small_Input/ISIC_' + str(item).rjust(7,'0') + '.jpg'
            img = cv2.imread(fileName)
        else:
            fileName = path + '/Small_GroundTruth/ISIC_' + str(item).rjust(7,'0') + '_segmentation.png'
            img = cv2.imread(fileName,0)
        X[i] = img
    if c == 1:
        X = numpy.swapaxes(X,1,3)
        X = numpy.swapaxes(X,2,3)    
    else:
        X = numpy.reshape(X,(X.shape[0],1,height,width))
    #X = X/255
    return(X)
    
# Performs prelimary operations on the input/output formats
def preliminary_processing(path):
    # Get file Names of input & output
    [inputNames,outputNames] = getFileNames(path,'/Small_Input/')
    # Splitting the data into training and testing set
    [inputNames_train, inputNames_test, outputNames_train, outputNames_test] = train_test_split(inputNames, outputNames, test_size=0.2)
    # Splitting the data into training and validation set
    [inputNames_train, inputNames_validation, outputNames_train, outputNames_validation] = train_test_split(inputNames_train, outputNames_train, test_size=0.1)
    
    # Gathering Data and Targets
    X_train = get_Images(inputNames_train,1)
    Y_train = get_Images(outputNames_train,2)
    X_test = get_Images(inputNames_test,1)
    Y_test = get_Images(outputNames_test,2)
    X_validation = get_Images(inputNames_validation,1)
    Y_validation = get_Images(outputNames_validation,2)
    #print(X_train)
    # Getting Training, Testing and Validation Set
    flags_train = numpy.ones(len(inputNames_train),dtype=int)
    train_set = SpatialDatasetNames(inputNames_train, flags_train, X_train, Y_train)
    flags_test = numpy.zeros(len(inputNames_test),dtype=int)
    test_set = SpatialDatasetNames(inputNames_test, flags_test, X_test, Y_test)
    flags_validation = numpy.zeros(len(inputNames_validation),dtype=int)
    validation_set = SpatialDatasetNames(inputNames_validation, flags_validation, X_validation, Y_validation)
    # Return Training, Testing & Validation Set
    return(train_set,test_set,validation_set)
    
def define_model_nParameters(device,batch_size,test_batch_size,train_set,validation_set,learning_rate,adam_weight_decay,lr_patience):
    # Defining the Data Loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=test_batch_size, shuffle=True, num_workers=8)
    # Defining the model
    model = UNet(1).to(device)
    #model = Net().to(device)
    # Defining the model parameters
    loss = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=adam_weight_decay)
    scheduler = ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=lr_patience,verbose=True,threshold=1e-6)
    return(model,loss,optimizer,scheduler,train_loader,validation_loader)
    
# Apply Exponential Contast as PreprocessingImageFolder(traindir, transform=transforms)
def apply_expContrast(img):
    gamma = 0.5 + random.random()
    invGamma = 1.0 / gamma
    table = numpy.array([((i / 255.0) ** invGamma) * 255 for i in numpy.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)
    return(img)
    
# Getting Images in Batches forboth Input & Output
def getBatchIO(data,target,t_flag):
    #path = '/media/heisenberg/8f00cc57-e994-44a3-98c8-a0df480082e7/SB_Works/BIPA Project/Segmentation'
    path = '/home/heisenberg/Desktop/SB_Works/Segmentation'
    data_size = data.size()[0]
    X = numpy.zeros((data_size,height,width,3),dtype=float)
    Y = numpy.zeros((data_size,height,width),dtype=float)
    fileNumber = data.numpy()
    for i, (item) in enumerate(fileNumber):
        input_fileName = path + '/Small_Input/ISIC_' + str(item).rjust(7,'0') + '.jpg'
        output_fileName = path + '/Small_GroundTruth/ISIC_' + str(item).rjust(7,'0') + '_segmentation.png'
        dim = (width,height)
        img = cv2.imread(input_fileName)
        #img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        if t_flag == 'train':
            choice = random.choice(list(range(9)))
            if choice != 9: img = apply_expContrast(img)
        #img = cv2.transpose(img)
        X[i] = img
        img = cv2.imread(output_fileName,0)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        #img = cv2.transpose(img)
        Y[i] = img
    Y = numpy.reshape(Y,(data_size,1,height,width))
    X = numpy.swapaxes(X,1,3)
    X = numpy.swapaxes(X,2,3)
    X = X/255
    Y = Y/255
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    return(X,Y)
    
def train(model, device, train_loader, optimizer, epoch,loss):
    # Training the model
    model.train()
    start_time = time.time()
    n_batches = len(train_loader)
    batch_print = 1
    running_loss = 0.0
    total_train_loss = 0
    for i, (imgName, data, target) in enumerate(train_loader):
        #[data,target] = getBatchIO(data,target,'train')
        data  = Variable(data.to(device=device, dtype=torch.float))
        target = Variable(target.to(device=device, dtype=torch.float))
        optimizer.zero_grad()
        output = model(data)
        output = torch.sigmoid(output)
        loss_amount = total_loss(output,target,loss)
        #loss_amount = loss(output,target)
        loss_amount.backward()
        optimizer.step()
        running_loss += loss_amount.item()
        total_train_loss += loss_amount.item()
        if (i + 1) % (batch_print) == 0:
            print("Epoch {}, {:.4f}% \t train_loss: {:.4f}".format(epoch, (100 * (i+1) / n_batches), running_loss/batch_print))
            running_loss = 0.0
    total_train_loss /= len(train_loader)
    # Print Running Train Loss
    print('Train set: Average Running loss: {:.4f}'.format(total_train_loss))
    time_taken = format_time(time.time()-start_time)
    print('Batch Processing Time: ' + time_taken)
    
# Calculate the Per Pixel Accuracy using IoU
def calculate_PerPixelAccuracy(output,target):
    smooth = 1e-5
    threshold_val = Variable(torch.Tensor([0.5])).cuda()
    predicted = (output > threshold_val).float() * 1
    #true_area = target.sum(dim=(1,2,3)).float()
    intersection = (predicted * target).sum(dim=(1,2,3)).float()
    #intersection = predicted.eq(target.view_as(predicted)).sum(dim=(1,2,3)).float()
    cardinality = (predicted + target).sum(dim=(1,2,3)).float()
    #iou = ((2. * intersection + smooth) / (cardinality + smooth)).float()
    iou = ((intersection + smooth) / (cardinality - intersection + smooth)).float()
    #iou = ((2. * intersection + smooth) / ((2*3*height*width) + smooth)).float()
    #true_area_ratio = (( true_area + smooth) / (target + smooth).sum(dim=(1,2,3))).float()
    threshold_jaccard = Variable(torch.Tensor([0.65])).cuda()
    iou = ((iou > threshold_jaccard).float() * iou).float()
    return(iou.mean())
        
def test(dataPartitionLabel,model,device,test_loader,loss):
    # Evaluating the model
    model.eval()
    test_loss = 0
    data_length = 0
    accuracy = 0
    total_truth = 0
    # With no gradient
    with torch.no_grad():
        for i, (imgName, data, target) in enumerate(test_loader):
            #[data,target] = getBatchIO(data,target,'test')
            data  = Variable(data.to(device=device, dtype=torch.float))
            target = Variable(target.to(device=device, dtype=torch.float))
            output = model(data)
            output = torch.sigmoid(output)
            #print(output)
            #[acquired_value,true_value] = calculate_PerPixelAccuracy(output,target)
            #accuracy += acquired_value
            #total_truth += true_value
            accuracy += calculate_PerPixelAccuracy(output,target)
            loss_amount = total_loss(output,target,loss)
            #loss_amount = loss(output,target)
            test_loss += loss_amount.item()
    test_loss /= len(test_loader)
    accuracy_IOU = accuracy/len(test_loader)
    #total_truth /= len(test_loader)
    # Print Average Loss
    #print(dataPartitionLabel + ' set: Average loss: {:.4f}, IoU Predicted: {:.4f}, Actual Area: {:.4f}'.format(test_loss,accuracy_IOU,total_truth))
    print(dataPartitionLabel + ' set: Average loss: {:.4f}, IoU Predicted: {:.4f}'.format(test_loss,accuracy_IOU))
    return(test_loss,accuracy_IOU)
    
def learning(model,device,loss,optimizer,scheduler,n_epochs,train_loader,validation_loader,es_patience):
    # Container for storing loss values per epoch
    train_loss_list = []
    validation_loss_list = []
    # Container for storing per pixel accuracy values per epoch
    train_accuracy_list = []
    validation_accuracy_list = []
    # Training Begins
    validation_decrease_counter = 0
    old_validation_loss = 100
    training_start_time = time.time()
    print('Neural Training Begin...')
    for epoch in range(1, n_epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss)
        dataPartitionLabel = 'Train'
        [train_loss,train_accuracy] = test(dataPartitionLabel, model, device, train_loader, loss)
        dataPartitionLabel = 'Validation'
        [validation_loss,validation_accuracy] = test(dataPartitionLabel, model, device, validation_loader, loss)
        if validation_loss <= old_validation_loss:
            validation_decrease_counter = 0
            torch.save(model,"checkpoint_Model.pt")
            old_validation_loss = validation_loss
        else:
            validation_decrease_counter += 1
        if es_patience <= validation_decrease_counter:
            model = torch.load("checkpoint_Model.pt")
            print('Validation error is increasing. Stopping Model.')
            break
        train_loss_list.append(train_loss)
        validation_loss_list.append(validation_loss)
        train_accuracy_list.append(train_accuracy)
        validation_accuracy_list.append(validation_accuracy)
        scheduler.step(train_loss)
    # Training Ends
    training_end_time = time.time()
    training_time = training_end_time - training_start_time
    time_taken = format_time(training_time)
    return(model,train_loss_list,validation_loss_list,train_accuracy_list,validation_accuracy_list)
    
# Plot Losses
def plot_losses(train_losses,validation_losses):
    itr = list(range(1,len(train_losses)+1))
    plt.plot(itr,train_losses,color='red',label ='Training Loss')
    plt.plot(itr,validation_losses,color='blue',label ='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc=1)
    plt.title('Loss with Epochs')
    plt.tight_layout()
    plt.savefig('loss.jpg')
    
# Plot IoUs
def plot_IoU(train_IoU,validation_IoU):
    itr = list(range(1,len(train_IoU)+1))
    plt.plot(itr,train_IoU,color='red',label ='Training IoU')
    plt.plot(itr,validation_IoU,color='blue',label ='Validation IoU')
    plt.xlabel('Epochs')
    plt.ylabel('IoU')
    plt.legend(loc=1)
    plt.title('IoU with Epochs')
    plt.tight_layout()
    plt.savefig('iou.jpg')
    
# Predict output Image
def predict_image(output):
    threshold_val = Variable(torch.Tensor([0.5])).cuda()
    predicted = (output > threshold_val).float() * 1
    return(predicted)
    
# Get all images in a figure
def print_Image(fileName,i_img,t_img,p_img,d_img):
    plt.figure(1)
    plt.subplot(2,2,1)
    plt.imshow(i_img)
    plt.title('Input Image')
    plt.subplot(2,2,2)
    plt.imshow(t_img)
    plt.title('Target Image')
    plt.subplot(2,2,3)
    plt.imshow(p_img)
    plt.title('Predicted Image')
    plt.subplot(2,2,4)
    plt.imshow(d_img)
    plt.title('Difference Showcased')
    plt.tight_layout()
    plt.savefig(fileName)

# Save the compared Images
def saveImages(fileName,data,target,predicted):
    #path = '/media/heisenberg/8f00cc57-e994-44a3-98c8-a0df480082e7/SB_Works/BIPA Project/Segmentation'
    path = '/home/heisenberg/Desktop/SB_Works/Segmentation'
    fileName = fileName.numpy()
    data = (data.cpu()).numpy()
    target = (target.cpu()).numpy()
    predicted = (predicted.cpu()).numpy()
    for i, (item) in enumerate(fileName):
        savepath = path + '/TestOutput/ISIC_' + str(item).rjust(7,'0') + '.jpg'
        inputImage = data[i]
        targetImage = target[i]
        predictedImage = predicted[i]
        imageDifference = ((targetImage-predictedImage)*255)
        temp1 = imageDifference
        temp2 = imageDifference
        temp1 = (numpy.absolute((temp1>0)*temp1)).astype(numpy.uint8)
        temp2 = (numpy.absolute((temp2<0)*temp2)).astype(numpy.uint8)
        temp3 = numpy.zeros((1,temp1.shape[1],temp1.shape[2]),dtype=numpy.uint8)
        imageDifference = numpy.vstack((temp1,temp2,temp3))
        imageDifference = numpy.swapaxes(imageDifference,0,2)
        imageDifference = numpy.swapaxes(imageDifference,0,1)
        #imageDifference = numpy.repeat(imageDifference[:, :, :, numpy.newaxis], 3, axis=3)
        inputImage = (inputImage*255).astype(numpy.uint8)
        targetImage = (targetImage*255).astype(numpy.uint8)
        targetImage = numpy.repeat(targetImage[:, :, :, numpy.newaxis], 3, axis=3)
        predictedImage = (predictedImage*255).astype(numpy.uint8)
        predictedImage = numpy.repeat(predictedImage[:, :, :, numpy.newaxis], 3, axis=3)
        targetImage = numpy.reshape(targetImage,(height,width,3))
        predictedImage = numpy.reshape(predictedImage,(height,width,3))
        #imageDifference = numpy.reshape(imageDifference,(height,width,3))
        inputImage = numpy.swapaxes(inputImage,0,2)
        inputImage = numpy.swapaxes(inputImage,0,1)
        print_Image(savepath,inputImage,targetImage,predictedImage,imageDifference)
    
def final_test(model,device,test_set,test_batch_size,loss):
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=True, num_workers=8)
    model.eval()
    test_loss = 0
    data_length = 0
    accuracy = 0
    total_truth = 0
    with torch.no_grad():
        for i, (imgName, data, target) in enumerate(test_loader):
            #[data,target] = getBatchIO(data_name,target,'test')
            data  = Variable(data.to(device=device, dtype=torch.float))
            target = Variable(target.to(device=device, dtype=torch.float))
            output = model(data)
            output = torch.sigmoid(output)
            #[acquired_value,true_value] = calculate_PerPixelAccuracy(output,target)
            #accuracy += acquired_value
            #total_truth += true_value
            accuracy += calculate_PerPixelAccuracy(output,target)
            predicted = predict_image(output)
            loss_amount = total_loss(output,target,loss)
            #loss_amount = loss(output,target)
            test_loss += loss_amount.item()
            saveImages(imgName,data,target,predicted)
    test_loss /= len(test_loader)
    accuracy_IOU = accuracy/len(test_loader)
    #total_truth /= len(test_loader)
    #print('Test set: Average loss: {:.4f}, IoU: {:.4f}, Actual Area: {:.4f}'.format(test_loss,accuracy_IOU,total_truth))
    print('Test set: Average loss: {:.4f}, IoU: {:.4f}'.format(test_loss,accuracy_IOU))

# Define Dice Loss    
def jaccard_loss(predicted,target):
    smooth = 1e-5
    intersection = (predicted * target).sum(dim=(1,2,3)).float()
    cardinality = (predicted + target).sum(dim=(1,2,3)).float()
    #loss = (1- ((2. * intersection + smooth) / (cardinality + smooth))).float()
    loss = (1- ((intersection + smooth) / (cardinality - intersection + smooth))).float()
    #loss = (1- ((2. * intersection + smooth) / ((predicted.size()[1] + predicted.size()[2] + predicted.size()[3]) + (target.size()[1] + target.size()[2] + target.size()[3]) + smooth))).float()
    return(loss.mean())

# Define Customize Loss
def total_loss(output,target,model_bce_loss):
    weigth_balance = 0.5
    bce_loss = model_bce_loss(output,target)
    val_jaccard_loss = jaccard_loss(output,target)
    #loss = bce_loss * weigth_balance + val_jaccard_loss * (1 - weigth_balance)
    loss = bce_loss + val_jaccard_loss
    return(loss)

def model_segmentation():
    
    # Initializing useful information
    print('Starting the program...')
    random_seed(11,True)
    device = torch.device("cuda")
    #path = '/media/heisenberg/8f00cc57-e994-44a3-98c8-a0df480082e7/SB_Works/BIPA Project/Segmentation'
    path = '/home/heisenberg/Desktop/SB_Works/Segmentation'
    
    # Get the training and Validation Datasets
    print('Fetching Filenames...')
    [train_set,test_set,validation_set] = preliminary_processing(path)
    
    # Defining the model Hyper-Parameters
    print('Defining Hyper-parameters...')
    batch_size = 2
    testing_batch_size = 2
    learning_rate = 2e-4
    n_epochs = 35
    adam_weight_decay = 1e-5
    lr_patience = 4
    es_patience = 6
    
    #Define the model, its parameters, the training and the validation data loaders
    print('Defining the model parameters...')
    [model,loss,optimizer,scheduler,train_loader,validation_loader] = define_model_nParameters(device,batch_size,testing_batch_size,train_set,validation_set,learning_rate,adam_weight_decay,lr_patience)
    
    # Learning of the Model
    [learned_model,train_losses,validation_losses,train_IoU,validation_IoU] = learning(model,device,loss,optimizer,scheduler,n_epochs,train_loader,validation_loader,es_patience)
    
    
    # Plot training and validation losses for each epoch
    print('Plotting Training and Validation Losses...')
    plot_losses(train_losses,validation_losses)
    
    # Plot training and validation IoU for each epoch
    print('Plotting Training and Validation IoUs...')
    plot_IoU(train_IoU,validation_IoU)
    
    # Converts Lists into Numpy Arrays and Saves them
    train_losses = numpy.asarray(train_losses)
    numpy.save('train_losses.npy',train_losses)
    validation_losses = numpy.asarray(validation_losses)
    numpy.save('validation_losses.npy',validation_losses)
    train_IoU = numpy.asarray(train_IoU)
    numpy.save('train_IOU.npy',train_IoU)
    validation_IoU = numpy.asarray(validation_IoU)
    numpy.save('validation_IOU.npy',validation_IoU)
    
    # Saving the learned Model
    print('Saving the model...')
    torch.save(learned_model,"learned_SkinLesion_Segmentation_Model.pt")
    
    #Final Testing of the Data
    print('Testing the Model on test data...')
    final_test(learned_model,device,test_set,testing_batch_size,loss)

width = 512
height = 384
        
path = '/home/heisenberg/Desktop/SB_Works/Segmentation'
model_segmentation()

