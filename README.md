# MDM

A Tensorflow implementation of the Mnemonic Descent Method.

    Mnemonic Descent Method: A recurrent process applied for end-to-end face alignment
    G. Trigeorgis, P. Snape, M. A. Nicolaou, E. Antonakos, S. Zafeiriou.
    Proceedings of IEEE International Conference on Computer Vision & Pattern Recognition (CVPR'16).
    Las Vegas, NV, USA, June 2016.

# Installation Instructions


## Menpo

We are an avid supporter of the Menpo project (http://www.menpo.org/) which we use
in various ways throughout the implementation.

Please look at the installation instructions at:

    http://www.menpo.org/installation/

## TensorFlow

Follow the installation instructions of Tensorflow at and install it inside the conda enviroment you have created

    https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#installing-from-sources

but use 

    git clone https://github.com/trigeorgis/tensorflow.git

as the TensorFlow repo. This is a fork of Tensorflow (#ff75787c) but it includes some
extra C++ ops, such as for the extraction of patches around the landmarks.

# Pretrained models

Disclaimer:
The pretrained models can only be used for non-commercial academic purposes.

A pretrained model on 300W train set can be found at: https://www.doc.ic.ac.uk/~gt108/theano_mdm.pb

# Training a model
Currently the TensorFlow implementation does not contain the same data augmnetation steps
as we did in the paper, but this will be updated shortly.

```
    # Activate the conda environment where tf/menpo resides.
    source activate menpo
    
    # Start training
    python mdm_train.py --datasets='databases/lfpw/trainset/*.png:databases/afw/*.jpg:databases/helen/trainset/*.jpg'
    
    # Track the train process and evaluate the current checkpoint against the validation set
    python mdm_eval.py --dataset_path="./databases/ibug/*.jpg" --num_examples=135 --eval_dir=ckpt/eval_ibug  --device='/cpu:0' --checkpoint_dir=$PWD/ckpt/train
    
    python mdm_eval.py --dataset_path="./databases/lfpw/testset/*.png" --num_examples=300 --eval_dir=ckpt/eval_lfpw  --device='/cpu:0' --checkpoint_dir=$PWD/ckpt/train
    
    python mdm_eval.py --dataset_path="./databases/helen/testset/*.jpg" --num_examples=330 --eval_dir=ckpt/eval_helen  --device='/cpu:0' --checkpoint_dir=$PWD/ckpt/train
    
    # Run tensorboard to visualise the results
    tensorboard --logdir==$PWD/ckpt
```


