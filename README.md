# TF_Activation

Library and analysis for [*Adversary Detection in Neural Networks via Persistent Homology*](http://arxiv.org/abs/1711.10056)

Apologies for the messiness of the repo. If you're looking for analysis w.r.t
the paper, the Jupyter Notebooks in `notebooks` is your best bet.

If you have specific questions, open an issue and I'll respond to it directly.
You can also send an email to the address listed in the paper.


## Installation:

Because the library incorporates a custom tensorflow operation via C++, it is 
required that Tensorflow be built from source on the host machine so that 
the custom operations can be compiled into the tensorflow package to be used 
by python. 

Because building Tensorflow from source is only supported on Mac and Linux, 
this package is also only supported on Mac and Linux.

#### Install Python3+

This package was developed with python 3.6, so python versions 3.+ will work. 

#### Build Tensorflow from source

Following the instructions here: 

https://www.tensorflow.org/install/install_sources. 

Google uses [Bazel](https://bazel.build/), so this will also need to be installed.

Also CUDA for the GPU endowed:

The following NVIDIA® software must be installed on your system:

  - GPU drivers. CUDA 9.0 requires 384.x or higher.
  - CUDA Toolkit (>= 8.0)
  - [cuDNN SDK (>= 6.0)](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)
  - CUPTI ships with the CUDA Toolkit, but you also need to append its path to the LD_LIBRARY_PATH environment variable: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64
