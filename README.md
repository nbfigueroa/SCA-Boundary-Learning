# SCA-Boundary-Learning
This repository contains the necessary libraries, scripts and instructions to learn a Dual-Arm Self-Collision Avoidance Boundary. With the following steps you can learn a Self-Collision-Avoidance Boundary for a Dual-Arm Maniupulator setup that is used as a constraint for a centralized inverse kinematic solver, as described in [1]:

<p align="center">
<img src="https://github.com/nbfigueroa/SCA-Boundary-Learning/blob/master/img/collision_nadia.gif" width="290"><img src="https://github.com/nbfigueroa/SCA-Boundary-Learning/blob/master/img/collision_sina.gif" width="290">
</p>

#### Reference
[1] [Sina Mirrazavi](http://lasa.epfl.ch/people/member.php?SCIPER=233855), [Nadia Figueroa](http://lasa.epfl.ch/people/member.php?SCIPER=238387) and Aude Billard, "A Unified Framework for Coordinated
Multi-Arm Motion Planning" *International Journal of Robotics Research* [In press]


## Step 1: Generate a Dual-Arm Collision Dataset

https://github.com/sinamr66/SCA_data_construction

Follow the instructions in the README file, you should modify the following input paramaters:
- Sampling resolution (joint angle increment, the 'resolution' variable is a multipler for 10 deg increments, i.e. resolution=2 gives a 20deg increment)
- Location of the robot bases wrt to each other
- DH parameters of the manipulators
- Joint workspace constraints, if any. 

This will generate a folder ./data which contains text files for the collided (y=-1) and non-collided (y=+1) joint configurations in form of the 3D positions of all joints wrt. one of the robot bases (the one defined as the origin x = [0 0 0]). This is the feature space that the SCA Boundary is learned in, if you have N joints (for all robots) your feature vector is ![alt text](https://github.com/nbfigueroa/SCA-Boundary-Learning/blob/master/utils/images_readme/CodeCogsEqn.gif "xinR").


## Step 2: Install dependencies for SCA-Boundary-Learning package
- [ML_toolbox](https://github.com/epfl-lasa/ML_toolbox): We use the cross-validation/grid-search functions + standard SVM learning (from libSVM libary).
- [SVMGrad](https://github.com/nbfigueroa/SVMGrad): SVMGrad is a compact library used to evaluate the decision function of a Gaussian RBF Kernel Support Vector Machine, as well as the its first Derivative (Gradient) in both MATLAB and C++.

## Step 3: Search for optimal hyper-parameters doing cross-validation with standard soft-margin SVM
Make sure to add all subfolders to your current directory and run the following script:
```
boundary_learning_robots.m
```
This will search for the optimal hyper-parameters that achieve the highest TPR and lowest FPR.

Note: The training stage in is not linearly increasing, i.e. a sample size of 12k point takes around 1 day, while a sample size of 24k take around 3 days.

## Step 4: Learn a sparse SVM via Cutting-Plane Training 
Using the optimal hyper-parameters learned from the previous step, you will now train a sparse SVM with a support vector budget K, this is set to default = 3,000. 

For a simple 2D example which can be visualized using SVMGrad run the following script:
```
boundary_learning_sparse_CSPS_2D.m
```
For a real robot dataset, run the following script:
```
boundary_learning_sparse_CSPS_robots.m
```
Once you've followed all the steps, you will have a MATLAB struct named cpsp_model_robots this contains all the parameters of the learnt SVM model:
- Support Vectors
- bias
- alpha*y
- gamma (i.e. inverse of Sigma)

Save the cpsp_model_robots struct AND some samples of the training (or testing) dataset for the next step.

## Step 5: To evaluate your learnt sparse SVM, we now generate a model for SVMGrad package and test it

https://github.com/nbfigueroa/SVMGrad

Follow the instructions in the README file. 

Must write instructions for this...

## Step 6: Test it on the real robots using the QP-IK-Solver testing package, this can be done in simulation.

https://github.com/sinamr66/QP_IK_solver

Must write instructions for this...

**Current Maintainer**: [Nadia Figueroa](http://lasa.epfl.ch/people/member.php?SCIPER=238387) (nadia.figueroafernandez AT epfl dot ch)

