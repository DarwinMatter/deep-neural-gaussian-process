This repository is a slight modification of the main repository of (1): NNGP: Deep Neural Network Kernel for Gaussian Process by Jaehoon Lee*, Yasaman Bahri*, Roman Novak, Sam Schoenholz, Jeffrey Pennington,
Jascha Sohl-dickstein

# Clone repository

To clone this repository, follow the following steps:

git clone https://github.com/DarwinMatter/deep-neural-gaussian-process nngp

cd nngp

docker build -t nngp-project .

docker run nngp-project

# Target figures:

**Figure 4:**

By following the steps in Lecture 9, we can generate figure 4 by using uncertainty.py. 

**Figure 3:**

In addition, we generate Figure 3 by using run_experiment.py. To generate Figure 3, we create  the file heat_map_figure.py. This file was created using ChatGPT. This script takes parameters weight_var, bias_var, num_train, and num_eval. The values of weight_var, and bias_var are used to create a grid. For each pair (weight_var, and bias_var), it runs the file run_experiments.py and produces an accuracy value. After generating accuracy values for each element in the grid, it generates an image called heat_map_intermediate_nonlinearityfunction.png, where nonlinearityfunction = relu, tanh, or sigmoid. To run this function, we follow the following steps:

1. In the terminal, go to the folder containing all the files (nngp).

2. Then update the docker created in Step 2 by using docker build -t nngp-project .

3. Then run docker run --platform linux/amd64 -it --entrypoint bash nngp-project to access docker.

4. Then, to generate the image, run (in this example nonlinearityfunction = relu and deep=50)

python heat_map_figure.py  
--weight_vars=0.5,0.9,1.3,1.7,2.1,2.5,2.9,3.3,3.7,4.0  --bias_vars=0.0,0.3,0.6,0.9,1.2,1.5,1.8,2.0 --depth=50 

--nonlinearity=relu 

--num_train=100 

--num_eval=1000 

--output_file=/nngp/heat_map_intermediate_relu.png

This image is saved inside the docker container as “heat_map_intermediate_relu.png”.

6. In order to export the image to the computer, one have to exit the docker and copy the image to the computer as follows:

1) Inside docker: exit

2) On the folder containing all the files, we run:

docker ps -a #To check the container ID where the image was stored

docker cp <CONTAINER-ID>:/nngp/heat_map_intermediate_relu.png . 	#To copy the image from the docker container to the computer

If one wants to generate the graph associated with the nonlinear function “tanh” or “sigmoid”, then we just have to change  nonlinearity=tanh or sigmoid. 

# Description of results and limitations

In our case, we generated three heat maps: one for relu, tanh, and sigmoid. In order to avoid memory limitation issues, we consider a smaller grid than the one used in the original paper. Moreover, to avoid memory or runtime issues when computing the F matrix, we consider the following modification: as stated in the paper, item 2. page 5, the construction of the matrix F “involves numerically approximating a Gaussian integral, in terms of the marginal variances s and correlations c”. In order to avoid this, we use the next values for gaussian integration grid, and variances and correlations grid: 

        ‘--n_gauss=101', 	default value =501
        
        '--n_var=151', 		default value =501
        
        '--n_corr=131', 		default value =500
    
which are smaller compared with the default values. 

As shown in the images (see heat_map_intermediate_relu and heat_map_intermediate_tanh), the heat maps for both relu and tanh with deep=50 are similar to those in the paper, although our images are less precise and coarser than those in the original paper because of our choices of grid size and parameters described above.

In the case of the sigmoid function (see heat_map_intermediate_sigmoid), when deep=50, the accuracy values in the heat map are very low compared with those associated with relu and tanh. However, when we change to deep=3 (see heat_map_intermediate_sigmoiddeep3), the accuracy of the model increases significantly, from 11.6% for deep=50 to 69% for deep=3. In fact, we can see that the heat map for deep=3 suggests that weight variance values close to 4 and bias variance values between 1.2 and 2 give better accuracy. Hence, NNGP with the sigmoid function seems to perform better with a small number of layers. This contrasts with the behavior of NNGP with relu and tanh, which, as can be seen in the graphs, perform poorly when using deep=3.

# References:

1) Jaehoon Lee, Yasaman Bahri, Roman Novak, Sam Schoenholz, Jeffrey Pennington, Jascha Sohl-dickstein. "Deep neural networks as gaussian processes." International Conference on Learning Representations (2018). https://openreview.net/forum?id=B1EA-M-0Z.
2) Lee, J., Bahri, Y., & Novak, R. (2018)a. NNGP: Deep neural network kernel for Gaussian process [Computer software]. GitHub. https://github.com/brain-research/nngp

