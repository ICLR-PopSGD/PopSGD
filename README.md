# PopSGD

This folder provides the code used to run the experiments on the Piz Daint supercomputer for the ***PopSGD: Decentralized Stochastic Gradient Descent in the Population Model*** submission for ICLR2020. 


## Running the code
First clone this repo to your desired location. 

`git clone https://github.com/ICLR-PopSGD/PopSGD.git`

Afterwards after setting up all the necessary libraries such as torch, torchvision, mpi4py, ... you run the code like so:

`mpiexec -n <workers> python worker.py --dataset-name imagenet --lr 1 --epochs 20 --log-interval 10 --save-model` 