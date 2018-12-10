# ResNet ImageNet Code

This repository provides code for both training and using the restricted robust resnet models from the paper: 

**Robustness May Be at Odds with Accuracy**  
*Dimitris Tsipras\*, Shibani Santurkar\*, Logan Engstrom\*, Alexander Turner, Aleksander Madry*  
https://arxiv.org/abs/1805.12152  

We show that adversarial robustness often inevitablely results in accuracy loss. The silver lining: adversarial training induces more semantically meaningful gradients and gives adversarial examples with GAN-like trajectories:  

<img src = 'https://pbs.twimg.com/media/DengXkCWAAADnUN.jpg:large' width='800px' />

### General overview ###

Before trying to run anything: 

- Run `./setup.sh`. 
- Get a downloaded version of the ImageNet training set. By default the code looks for this directory in the environmental variable `INET_DIR`. For example, run `export INET_DIR=/scratch/datasets/imagenet/`. 

The code is organized so that you can: 

- Train your own robust restricted ImageNet models (via `ez_train.sh`)
- Produce adversarial examples and visualize gradients, with example code in `example_use.ipynb`
- Reproduce the ImageNet examples seen in the paper (via `make_viz.sh`). 

### Pretrained models
This repository comes with (after following the instructions) three restricted ImageNet pretrained models:

- `data/train_224_nat_slim` corresponds to a naturally trained model
- `data/train_224_robust_eps_0.005_lp_inf_slim` corresponds to an linf adv trained model with eps 0.005
- `data/train_224_robust_eps_1.0_lp_2_slim` corresponds to an l2 adv trained model with eps 1.0

You will need to set the model ckpt directory in the various scripts/ipynb files where appropriate if you want to complete any nontrivial tasks. 
