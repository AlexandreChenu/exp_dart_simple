# Individual Research Project - Controlling Robots with Neural-Networks - First Experience - Hexapod Robot 

This repository contains all the code required to run the first experience of Alexandre Chenu's Individual Research Project 
applied to a Hexapod Robot. The objective is to learn both high-performing and diverse controllers for a Hexapod Robot to reach 
a fixed point in the work space. In this experience, an archive of approximately 1000 Neural Networks controllers is learned 
using a Quality-Diversity algorithm. 

## Dependencies

* Ubuntu
* DART simulator, http://dartsim.github.io/ (release-6.1 branch)
* Eigen 3, http://eigen.tuxfamily.org/
* Boost library
* SFERES2 library, https://github.com/sferes2/sferes2 (qd branch)
* NN2 additional module for SFERES2, https://github.com/sferes2/nn2

## Models

models/ contains several evolved Neural Network based controllers.  

* model_25000.bin is trained to reach a target positioned in coordinates (-0.5,0.5)
* model_25000-031.bin is trained to reach a target position in coordinates (1,0.5)

## How to run the demo

1. Clone the repository.

```
git clone https://github.com/AlexandreChenu/exp_dart_simple.git
```

2. Move to singularity repository.

```
cd exp_dart_simple/singulariy
```

3. Build the demonstration singularity image.

```
./build_final_image.sh
```

This should create a new executable final_exp_dart_simple_DATE_TIME.sh.

4. Run the demonstration

```
./final_exp_dart_simple_DATE_TIME.sh
```

## How to modify code and experiences

All the code contained in this repository can be modified in order to adapt it and continue its development. 
All dependencies are installed and pre-compiled on a singularity container. Therefore, you only need to create a sandbox 
container using the two following commands. 

1. Move to singularity container 
```
cd exp_dart_simple/singulariy
```

2. Build sandbox container 
```
./start_container.sh
```

After editing the code, you must compile it using file. Please refer to https://gitlab.doc.ic.ac.uk/AIRL/AIRL_WIKI/wikis/how-to-use-AIRL_environment-and-create-you-own-experience
for more information about the environment and how to use it. 

## How to modify singularity containers

Singularity containers may be personalized by editing the singularity.def file. 
```
vim singularity.def
```

## Material

Here is a quick summary of all the main files contained in this repository. 

* wscript - waf script for compilation
* test_dart - main file for running Quality-Diversity algorithm (contains includes, parameters, template-type declaration..)
* best_fit_nn - redefinition of bestfit.hpp from sferes::dart (saves the best model contained in the archive)
* best_fit_all - redefinition of bestfit.hpp from sferes::dart (saves all the model contained in the archive at the end of the evolution)
* best_fit_nov - redefinition of bestfit.hpp from sferes::dart (saves the novelty score of the archive)
* fit_hexa_control_nn - definition of the evaluation step
* hexa_control.hpp/hexa_controller_simple.hpp/policy_control - redefinition of DART library files to include NN2 neural-networks
* desc_hexa - descriptor file for interacting with the robot during the learning process
* gen_mlp - definition of a new Feed-Forward Neural Network genotype
* test_model_dart - testing file for visualizing the hexapod

## Contributors

* Alexandre Chenu 
* Antoine Cully
* Szymon Brych
