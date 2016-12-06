---
layout: post
title: How to Install Theano on Windows 10 64b to try deep learning on GPUs
---

[Deep learning](http://en.wikipedia.org/wiki/Deep_learning) is hot! Mostly due to significantly improved [results](http://www.technologyreview.com/s/513696/deep-learning/) that you might have heard about. The use of graphical processing units ([GPUs](http://en.wikipedia.org/wiki/Graphics_processing_unit)) that can perform many calculations in parallel has been instrumental to these advancements. Nvidia has been a pioneer in this and you can try deep learning on their GPUs with the least hassle! The concepts behind deep learning are actually very simple and algorithms are more than two decades old. They are just surprisingly easy to understand conceptually if you know simple linear algebra and math although there are many tricks to make them work better. Check out Chris Olah's [blog](http://colah.github.io/) for more on that. What's different today from say 20 years ago then? Well, mostly there is tons of very large structured datasets available and we've figured out we can use GPUs. So if you want to get a taste of deep learning for yourself what you need to do is to try training neural networks with GPUs to get a sense of how things work. You don't even need a fancy GPU to get started. My old laptop has an old [GeForce GT 630M GPU](http://www.geforce.com/hardware/notebook-gpus/geforce-gt-630m) and it still gives me a roughly 8-10 times faster computation on simple neural networks than my core i7 CPU.

[theano](http://en.wikipedia.org/wiki/Theano_%28software%29) is a great library to get started with that provides the tools you need to build and train a simple neural network in python. If you google you can find good tutorials online that can guide you through building and training a deep NN. However, before you start you need to install theano. If you are using Windows it might be a little challenging to get theano up and running. A little while back I tried doing this and after a lot of googling and trial and error, this is the process I used to get it to work. So I thought I'd share it here with those interested.

1) Download Anaconda for Windows x64 for Python 2.7 (Don`t use the Python 3.5 it will not work!)

[https://www.continuum.io/downloads](http://www.continuum.io/downloads)

2) After anaconda installation open a command prompt and execute:

```
conda install pip six nose numpy scipy

conda install mingw libpython
```

3) Clone the theano project to your local machine from github. (I assume you know how github works if not check [https://www.youtube.com/watch?v=0fKg7e37bQE](http://www.youtube.com/watch?v=0fKg7e37bQE))

```
git clone https://github.com/Theano/Theano.git
```

4) Open a command prompt and navigate to the theano project folder and execute:

```
python setup.py install
```

You can test your installation by creating a test.py file with the following code line:

```
import theano
```

In a command prompt navigate to the folder containing the test.py file and execute:

```
python test.py
```

5) I encountered the error: "no module named nose_parameterized". Installing nose-parameterized solved it.

```
pip install nose-parameterized
```
At this point your should have a functioning installation of theano. You can test it by opening a python console and typing in

```
import theano
```
if you want performance, there are two more things you need to do. If you want fast CPU calculations you will need to install a BLAS library. Alternatively, you can have theano use your GPU.

make a new environment variable named THEANO_FLAGS with following value:
```
floatX=float32,device=cpu
```

## Fast CPU computing:

6)OpenBlas for fast CPU computing:

-   Download the precompiled libopenblas.dll from [openblas/v0.2.14/](http://sourceforge.net/projects/openblas/files/v0.2.14/)
-   Download [mingw64_dll.zip](http://sourceforge.net/projects/openblas/files/v0.2.14/mingw64_dll.zip/download) as well

I chose [OpenBLAS-v0.2.14-Win64-int32.zip](http://sourceforge.net/projects/openblas/files/v0.2.14/OpenBLAS-v0.2.14-Win64-int32.zip/download).

next extract the dll in bin folder of the zip to C:\openblas, extract all dll's in mingw64 to the same location

finally set blas.ldflags=-LC:\\openblas -lopenblas in THEANO_FLAGS after previous values seperated by a comma.

## GPU Computing:

7) Install Visual Studio 2013 Community Edition (VS 2015 DOES NOT WORK WITH CUDA 7.5 AT THIS TIME)

8) Install CUDA Toolkit 7.5

9) Add a "THEANO_FLAGS" environment variable with value "floatX=float32,device=gpu,nvcc.fastmath=True"

10) Add path to VS's C++ compiler (cl.exe) to your PATH environment variable, e.g. C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin

11)  install pyCUDA

Best way to install PyCUDA is to get the Unofficial Windows Binaries by Christoph Gohlke [here](http://www.lfd.uci.edu/%7Egohlke/pythonlibs/).

For quick setup, you can use [pipwin](http://github.com/lepisma/pipwin) which basically automates the process of installing Gohlke's packages.

```
pip install pipwin

pipwin install pycuda
```

12)make a .theanorc. (notice the . at the end) file in your home folder (mine is c:\users\hamid) and put the following inside:

```
[global]

floatX = float32

device = gpu

[blas]

ldflags=-LD:\\openblas -lopenblas

[gcc]

cxxflags=-ID:\OpenBLAS\include -LD:\OpenBLAS\lib

[nvcc]

fastmath = True

flags=-LD:\Anaconda\libs

compiler_bindir=D:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin
```

notice the paths for different directories on your machine.

12) Test theano on GPU using code from [theano docs](http://deeplearning.net/software/theano/tutorial/using_gpu.html)

Yay! Now you can train neural networks on your GPU!