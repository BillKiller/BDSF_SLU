## Boundary-prediction and Dynamic-template Slot Filling(BDSF)

## Preparation

Our code is based on PyTorch 1.1 and runnable for both windows and ubuntu server. Required python packages:
    
> + numpy==1.16.2
> + tqdm==4.32.2
> + scipy==1.2.1
> + torch==1.1.0
> + ordered-set==3.1.1

We highly suggest you using [Anaconda](https://www.anaconda.com) to manage your python environment.

## How to Run it

The script **train.py** acts as a main function to the project. For reproducing the results reported in our
paper, We suggest you the following hyper-parameter setting for ATIS dataset:

        python train.py -wed 256 -ehd 256 -aod 128 

Similarly, for SNIPS dataset, you can also consider the following command: 

        python train.py -wed 32 -ehd 256 -aod 128

Due to some stochastic factors, It's necessary to slightly tune the hyper-parameters using grid search. If you have any question, please issue the project or email [me](yangmingli@ir.hit.edu.cn) and we will reply you soon.
