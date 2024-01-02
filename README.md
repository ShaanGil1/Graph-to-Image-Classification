# Graphical Image Processing
Group project for CS 6220 - Fall 2023

The code for this project is broken up into two parts. Image-Graph-Conversion and then the graph classfication. 

The Image-Graph-Conversion is stored in the `image-graph-conversion` directory. The converters are broken up into 3 files: `AbstractConverter.py`, `BaseConverter.py`, `ImageConverter.py`. 

The `BaseConverter.py` stores the base abstract class used to create converter. It contains all the generic code to build the graphs besides finding the nodes and edges.

The `AbstractConverter.py` contains the converters that extend the Base Converter, but only implements the algorithm for creating the nodes.

The `ImageConverter.py` contains the converters that extened an AbstractConverter and implements the edge building code. 

This structure allows us to swap in and out different node and edge building algorithms interchangabley with modular architecture.

`build_dataset_ipynb` is the file we used to apply the converters to the dataset. To create a new dataset we would just change the converter and the output file name used.

`converter_demo.ipynb` is a file that we used to test and demonstrate the converter functionality internally.

The classification code is contained in the files outside of the `image-graph-conversion` directory. 

`resnet.py` contains the code we used to train and evaluate our resnet baseline. We also have a `resnet.ipynb` file, but we converted it to a py file so that we could run it on a linux cli rather than through juypter.

`graph_classification.ipynb` is the script we used for building and training our GNN models.

## Links to Open Source packages:

OpenCV: https://github.com/opencv/opencv

scikit-learn: https://github.com/scikit-learn/scikit-learn

pytorch geometric: https://github.com/pyg-team/pytorch_geometric

pytorch: https://github.com/pytorch/pytorch
Our Git Repo: https://github.gatech.edu/sjankowski6/graphical-image-processing 
