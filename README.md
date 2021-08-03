# Meibomian Glands Recognition

The project aims to recognize upper and lower meibomian glands area for a provided images' dataset.

## Material Preparation

To reproduce the results obtained, it is necessary to insert the material in the appropriate folders in which, in addition to the masks and images of the glands, there is also the pre-calculated classifiers and the dataset used to generate them.

## Prerequisites

* A development environment that supports Python3 is required

* We recommend to install **ThunderSVM**, in an environment that also supports CUDA, to perform classification using the GPU (and also to reuse the precomputed classifiers).

## Legend

`+` means any modifiable parameters.

`-` means any modifiable variables.
		
## Gabor Filter Parameters

Gabor filter parameters:

	+ num_theta: represents the orientation of the normal to the parallel stripes and, given that the shape is vertical in most cases, we have to use few orientation otherwise it could give a strong response even for lashes (default: 2).
	
	+ gabor_ker_size: that is a Gaussian kernel modulated by a complex harmonic function and it would be bigger to identify a wide area from the image (default: 129).
	
	+ lam: represents the wavelength of the sinusoidal factor (default: 1.01, 1.015).
	
	+ sigma: is the sigma/standard deviation of the Gaussian envelope (default: gabor_ker_size/5).
	
	+ psi: is the phase offset (default: π/2).
	
	+ gamma: is the spatial aspect ratio and specifies the ellipticity of the support of the Gabor function (default: 1.5).


### Code

`create_mask.py`: it allows to select the area of the gland by hand and generate the relative mask.
		 
	- Set the *isUpperGland* variable to True if you want to create masks for the upper glands, False otherwise.
	
	- Set the *create_mask_matrix* variable to True if you want to confirm the creation of the matrices containing the masks, False otherwise.
	
	- Set the *create_mask_image* variable to True if you want to confirm the creation of the masks images, False otherwise.
	
	+ n_segments: number of segments present in the image that can be clicked by hand and which allows us to generate the mask (default: 180).

`main.py`: it allows to create the covariance matrices that will be later split into train and test to generate the dataset.
		 
	- Set the *isUpperGland* variable to True if you want to create covariance matrices for the upper glands, False otherwise.
	
	- Set the *create_matrix* variable to True if you want to confirm the creation of the covariance matrices, False otherwise.
	
	- Modify the *name_file* variable if you want to change the resulting file name and to not overwrite the previous results.
	
	+ tile: size of the grid cells (default: 10).
	
	+ balance_dataset: percentage indicating the balance of negative/positive samples to avoid an unbalanced dataset (default: 30%).

`main-single.py`: it allows to create the covariance matrix of a single image, necessary if you want to predict the gland's area of an image:
	
	- Set the *isUpperGland* variable to True if you want to predict the upper gland area of ​​an image, False otherwise.
	
	- Set the *create_matrix* variable to True if you want to confirm the creation of the covariance matrix, False otherwise.
	
	- Modify the *name_file* variable if you want to change the resulting file name and to not overwrite the previous results.
	
	- Write the name of the image that we want to predict in the variable *name*. 
	
`train-gpu.py`: function that allows to save the SVM classifier that recognizes the area of ​​the glands starting from the covariance matrices (they form the train set):

	- Set the *selectGPU* variable according to the GPU to be used.
	
	- Set the *datasetNotCreate* variable to True if the train_test_split has not yet been done and therefore the dataset has not yet been generated from the covariance matrix, False otherwise.
	
	- Set the variable *isUpperGland* to True if you want to do the classification on upper glands images, False otherwise.
	
	- Set the *useBestParam* variable to True if you want to do the classification through the **GridSearchCV** function which allows to find the best parameters for the classifier, False otherwise.
	
	- Modify SVM kernel if you want to use a different one (default: *rbf*).
	
	- Modify the filename for the classifier so as not to overwrite the results obtained.
	
`predict.py`: function that allows to predict the glands' area of the test set and it generates as output a log file containing the accuracy obtained.

	- Set the *selectGPU* variable according to the GPU to be used.
	
	- Set the *isUpperGland* variable to True if you want to predict the upper glands, False otherwise.
	
	- Modify the name of the log file according to the prediction obtained.
	
`predict_refactor.py`: function that allows to predict the glands' area of a requested image and it generates as output the predicted image and the relative mask.

	- Set the *selectGPU* variable according to the GPU to be used.
	
	- Set the *isUpperGland* variable to True if you want to identify the upper gland area of ​​an image, False otherwise.
    
    	- Write the name of the image that we want to predict in the variable *name*.

### Utils
  
The *utils* folder contains the functions:
	
* `balance_dataset.py`: it contains the function that allows to launch the dataset.

* `folder_operations.py`: it contains the functions that allow to read the masks and images from the folders and create the masks.

* `gabor.py`: it contains the functions that allow to create gabor kernels and apply filters to the images.

* `matrix.py`: it contains the functions that allow to create and read '.mat' matrices.

* `read_save_images.py`: it contains the functions that allow to read, save and show images.

* `refactor_image.py`: it contains the function that allows to reconstruct the image from the features, that which recreates the original image from the cropped image and that which determines the coordinates of a block.

* `save_read_classifier.py`: it contains the functions that allow to read and save the classifiers.

* `split_image.py`: it contains the function that allows to divide the image into blocks and that which calculates the label of each block.

* `superpixel.py`: it contains the functions that allow to create the superpixels of an image and select the superpixels by hand.
