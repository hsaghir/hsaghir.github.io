---
layout: article
title: NLP
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---


## bash scripting
- A Bash script is a plain text file which contains a series of commands. These commands are a mixture of commands we would normally type ouselves on the command line
- convention is to give files that are Bash scripts an extension of .sh


### profile code to understand runtime bottlenecks

Profile your code with `sProfile` like the following command in order to understand which part of the code is taking how much time. 

```bash
python -m cProfile -o train_dialog_coherence.prof train_dialog_coherence.py --cuda --batch_size=16 --do_log --load_models --remove_old_run --freeze_infersent
```

After profiling, you can visualize it using `snakeviz`. But to run it on a remote machine, you need to deactivate running of browser and pipe the port to your local machine so that you can access the visulization in your local browser. On your remote machine do:

```bash
snakeviz train_dialog_coherence.prof -s --port=1539
```

to forward the port, make a pipe on your local:

```bash
ssh -A -N -f -L localhost:$local_host:localhost:$r_port -J skynet hamid@$remote_host
```

### Mac terminal colors

Open Terminal and type nano .bash_profile
Paste in the following lines:
```bash
export PS1="\[\033[36m\]\u\[\033[m\]@\[\033[32m\]\h:\[\033[33;1m\]\w\[\033[m\]\$ "
export CLICOLOR=1
export LSCOLORS=ExFxBxDxCxegedabagacad
alias ls='ls -GFh'
```

## pytorch

### Batching variable length sequences:
- There are two ways of batching variable length sequences:
    + Packing sequences of same size together in a minibatch and sending that into the LSTM, but that's not always possible. 
    + Padding sequnces with zero so that all have the same maximum seq-len size. This can be done two ways:
        * Simply feeding the padded sequences in a minibatch and get a fixed length of output. The desired output should have different lengths since sequences had different lengths and RNN should have unrolled only for the real sequences, so we have to mask them manually.
            - mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            - h_next = h_next*mask + hx[0]*(1 - mask)
            - c_next = c_next*mask + hx[1]*(1 - mask)
        * Using pytorch "pack_padded_sequence" which does a combination of packing and masking so that the output of each example will have different length. The RNN output then has to be unpacked using "pad_packed_sequence". 
            - Pad variable length sequences in a batch with zeros 
            - Sort the minibatch so that the longest sequence is at the beginning
            - pass the tensor and the lengths of sequences to "pack_padded_sequence"
            - the output of "pack_padded_sequence" goes to the RNN
            - RNN output is passed to "pad_packed_sequence" to map the output back to a zero-padded tensor corresponding to the right seq sizes.

- pack_padded_sequence removes padded zeros and packs data in a smaller tensor containing all contents of the minibatch. Instead, it makes the minibatch sizes variable for each time step. The RNN is still taking the maximum length number of steps, e.g. if the maximum sequence length is 35, the RNN will take 35 time steps for all the minibatches, but inside a minibatch,  there are different length samples. So pack_padded_sequence adapts the batch_size inside each minibatch for each time step to accomodate different length samples. 
    + For example, if maximum seq_legth is 6 and batch_size is 4, then each minibatch is a 6x4 tensor with zeros for shorter sequences than 6.
    + Imagine a minibatch with sample_lengths of 6, 5, 4, 3. The RNN needs to unroll for 6 time steps.  The first time step includes 4 samples, the second time step also 4 and so on i.e. [4, 4, 4, 3, 2, 1]. This means that all four words of each sequence will be fed into the LSTM at timestep 1. Then another 4 until the shorted sequence which was length 3 is exhausted. We then go on with 3 , 2, and then only the one word for the longest sequence of length 6.

### Pytorch Dataset class:
Pytorch has a dataset class that provides some tools for easy loading of data. 
    - **Dataset class** your dataset class should inherit **"torch.utils.data.dataset"**.  At this point the data is not loaded on memory. Several methods need to be implemented:
        + __init__(self) load and preprocess the data here or in __getitem__ for memory efficiency.
        + __len__(self) returns the size of the dataset.
        + __getitem__(self) indexes into dataset such that dataset[i] returns i-th sample.
    - **"torch.utils.data.DataLoader"** This class is used to get data in batches from the dataset class and provides an iterator to go through the data. It can shuffle and minibatch the data and load into memory. It has a default **collate_fn** that tries to convert the batch of data into a tensor but we can specify how exactly the samples need to be batched by implementing **collate_fn**. Usage in training is as simple as instantiating the Dataloader class with the dataset instance and a for loop with **enumerate(dataloader)**.
    - **DataParallel**: Data Parallelism is when we split the mini-batch of samples into multiple smaller mini-batches and run the computation for each of the smaller mini-batches in parallel. One can simply wrap a model module in DataParallel and it will be parallelized over multiple GPUs in the batch dimension.

### using Tensorboard with pytorch or python in general
It's actually very easy to use tensorboard anywhere with python. 
```python
from tensorboard_logger import Logger as tfLogger
logdir = '/tmp/tb_files'
tflogger = tfLogger(logdir)
tflogger.scalar_summary('Training Accuracy', train_acc, epoch)
```

then run tensorboard server on your remote machine (similar to jupyter server)
```bash
tensorboard --logdir=/tmp/tb_files  --port=8008
```

And connect to it by tunneling the port to your local machine. Run is on your local machine and then visin tensorboard in browser at [](http://localhost:7003/)

```bash
ssh -A -N -f -L localhost:7003:localhost:8008 -J skynet hamid@compute006
```

## Debugging ML code:
- start with a small model and small data and evolve both together. If you can't overfit a small amount of data you've got a simple bug somewhere. 
    + Start with all zero data first to see what loss you get with the base output distribution, then gradually include more inputs (e.g. try to overfit a single batch) and scale up the net, making sure you beat the previous thing each time.
    + also if zero inputs produces a nice/decaying loss curve, this usually indicates not very clever initialization.
    + initialize parameters with truncated normal or xavier.
    + also try tweak the final layer biases to be close to base distribution
    + for classification, check if the loss started at ln(n_classes)

- remember to toggle train/eval mode for the net. 
- remember to .zero_grad() (in pytorch) before .backward(). 
- remember not to pass softmaxed outputs to a loss that expects raw logits.
- pytorch `.view()` function reads from the last dimension first and fills the last dimension first too
- when comparing tensors, the results are `ByteTensor`s. ByteTensors have a buffer of `255` after which it is zeroed out. Although this issue seems to be fixed in newer pytorch versions, beware that a `sum()` on ByteTensors is likely to result in wrong answer. First convert them to `float()` or `long()` and then `sum()`



If your network isn’t learning (meaning: the loss/accuracy is not converging during training, or you’re not getting results you expect), try these tips:

Overfit! The first thing to do if your network isn’t learning is to overfit a training point. Accuracy should be essentially 100% or 99.99%, or an error as close to 0. If your neural network can’t overfit a single data point, something is seriously wrong with the architecture, but it may be subtle. If you can overfit one data point but training on a larger set still does not converge, try the following suggestions.
Lower your learning rate. Your network will learn slower, but it may find its way into a minimum that it couldn’t get into before because its step size was too big. (Intuitively, think of stepping over a ditch on the side of the road, when you actually want to get into the lowest part of the ditch, where your error is the lowest.)
Raise your learning rate. This will speed up training which helps tighten the feedback loop, meaning you’ll have an inkling sooner whether your network is working. While the network should converge sooner, its results probably won’t be great, and the “convergence” might actually jump around a lot. (With ADAM, we found ~0.001 to be pretty good in many experiences.)
Decrease (mini-)batch size. Reducing a batch size to 1 can give you more granular feedback related to the weight updates, which you should report with TensorBoard (or some other debugging/visualization tool).
Remove batch normalization. Along with decreasing batch size to 1, doing this can expose diminishing or exploding gradients. For weeks we had a network that wasn’t converging, and only when we removed batch normalization did we realize that the outputs were all NaN by the second iteration. Batch norm was putting a band-aid on something that needed a tourniquet. It has its place, but only after you know your network is bug-free.
Increase (mini-)batch size. A larger batch size—heck, the whole training set if you could—reduces variance in gradient updates, making each iteration more accurate. In other words, weight updates will be in the right direction. But! There’s an effective upper bound on its usefulness, as well as physical memory limits. Typically, we find this less useful than the previous two suggestions to reduce batch size to 1 and remove batch norm.
Check your reshaping. Drastic reshaping (like changing an image’s X,Y dimensions) can destroy spatial locality, making it harder for a network to learn since it must also learn the reshape. (Natural features become fragmented. The fact that natural features appear spatially local is why conv nets are so effective!) Be especially careful if reshaping with multiple images/channels; use numpy.stack() for proper alignment.
Scrutinize your loss function. If using a complex function, try simplifying it to something like L1 or L2. We’ve found L1 to be less sensitive to outliers, making less drastic adjustments when hitting a noisy batch or training point.
Scrutinize your visualizations, if applicable. Is your viz library (matplotlib, OpenCV, etc.) adjusting the scale of the values, or clipping them? Consider using a perceptually-uniform color scheme as well.