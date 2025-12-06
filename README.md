This project builds a simple Neural Network using the simoid activation function and the Mean Squared Error as the cost function to detect the correct number on a given image of a handwritten digit taken from the MNIST dataset. The aim of this project was to learn and implenment the algorithms behind a Neural Net using only the basic python functions and no libraries such as scikit-learn. 


# User Guide

## To run the program:

First clone the repository into your local device and install the poetry dependencies (already defined in the `pyproject.toml` file) by running

```bash
poetry install --no-root
```

You can either train the model yourself or use the pretrained model given as the pickle file in `trained_model.pkl`


To train the model yourself simply run the following file: `src/train_model.py`. The model currently trains on the handwritten digits MNIST image dataset. It accepts a list of one dimensional vectors which are the grey scale image matrices dropped into single vectors. If you want to use a different dataset, you will have to modify the number of input neurons based on how many pixels are in each image of your custom dataset, and feed the network the custom dataset in the correct format. 


To use the pretrained model you can follow the example notebook I’ve created in `src/results.ipynb`


For understanding what the actual algorithm does, which is defined in `src/architecture.py` , you can check out the file `src/equations_explained.markdown`, which briefly explains the mathematics behind all of the equations used in the algorithm. 
