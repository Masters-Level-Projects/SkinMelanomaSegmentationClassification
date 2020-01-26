The classification code consists of three code files:

preparedata3.py - it arranges the given dataset into a format the model may consume. Also, the dataset is split into train, validation and testset.

cnn_pytorch3.py - the model for classification.

tester.py - perform testing on test dataset and generate the final confusion matrix for test set

------------------------------------------------
To prepare the dataset, place the directory containing the given dataset, in the same directory and then execute the following:

python preparedata3.py

To run the training process, execute the model file at a terminal at a path containing the dataset prepared as above:

python cnn_pytorch3.py

The above will automatically call the tester  module and generate the confusion matrix.

In case testing needs to be done separately, run the tester module as follows:

python tester.py
