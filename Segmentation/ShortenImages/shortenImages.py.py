import cv2
import numpy
import os
from multiprocessing import Pool


# Perform Shorten Operation
def shorten_images(inputPath,outputPath):
    
    # Predefined Width and Height
    width = 512
    height = 384
    
    dim = (width,height)
    img = cv2.imread(inputPath)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite(outputPath,img)


# Produce and Save Smaller Dimension Images
def produce_Shorter_Images(path):
    
    # Set number of workers for multiprocessing
    number_of_workers = 8
    
    # Large Images path
    inputDataFolderName = 'Input/'
    inputTargetFolderName = 'GroundTruth/'
    inputDataPath = path + inputDataFolderName
    inputTargetPath = path + inputTargetFolderName
    
    # Smaller Output Path
    outputDataFolderName = 'Small_Input/'
    outputTargetFolderName = 'Small_GroundTruth/'
    outputDataPath = path + outputDataFolderName
    outputTargetPath = path + outputTargetFolderName
    
    # Save Smaller Images for Data
    print('Decreasing Dimensions of Input')
    namelist = os.listdir(inputDataPath)
    parameter_tuples = [(inputDataPath+namelist[i],outputDataPath+namelist[i]) for i in range(len(namelist))]
    with Pool(number_of_workers) as p: p.starmap(shorten_images,parameter_tuples)
        
    # Save Smaller Images for Target
    print('Decreasing Dimensions of Target')
    namelist = os.listdir(inputTargetPath)
    parameter_tuples = [(inputTargetPath+namelist[i],outputTargetPath+namelist[i]) for i in range(len(namelist))]
    with Pool(number_of_workers) as p: p.starmap(shorten_images,parameter_tuples)


path = '/media/heisenberg/8f00cc57-e994-44a3-98c8-a0df480082e7/SB_Works/BIPA Project/Segmentation/ShortImages/'
produce_Shorter_Images(path)
