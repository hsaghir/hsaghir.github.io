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