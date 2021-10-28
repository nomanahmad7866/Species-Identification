# Birds Species-Identification (Bs Research)

In this research, the primary goal was to make the network more accurate with less computation using different optimization techniques including filter pruning and quantization. I believe the blend of different techniques on a single network/model can be helpful in this regard.
Following technique is used to accomplish state of the art results:

1)	Caltech-UCSD Birds 200 (CUB-200) was taken for training and evaluation  purposes
2)	Training data was preprocessed in order to remove unwilling image distortions which enhance image features result in lack of precision.  
3)	Data augmentation technique was used to create modified version of images to improve performance and ability of model to generalize.
4)	The Resnet model was used by the courtesy of transfer learning and trained.  
5)	Finally, the proposed model was successfully tested on unseen images of 200  different birds categories.
6)	At the end of this process, 69% accuracy was obtained on the test dataset

Dataset:
         Open Source Caltech-UCSD-200 dataset is used which is preprocessed to achieve desire results. the dataset can be accessed 
         [http://www.vision.caltech.edu/visipedia/CUB-200.html] 
         


