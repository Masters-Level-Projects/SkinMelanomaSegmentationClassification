This is the ReadMe to run the segmentation portion.
We implemented three models, which are given in three folders: Model 1, Model 2 and Model 3.
But to train the model, the raw Images needs to downloaded from the dataset link, and the input
images needs to be supplied to the Input folder inside folder ShortenImage, and the segmented
Images needs to be supplied to the GroundTruth folder. The code shortenImages.py needs to be
run, and these shortened Images needs to be supplied to the Model's folder.
The Model 3 was run on GoogleColab, thus, minor path changes needs to be done, before running
the code properly.
The model is getting saved, in the same folder, and could be run for post training purposes.
The training & validation losses & accuracies, are being saved in the model's folder only.