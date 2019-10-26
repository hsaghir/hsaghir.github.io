---
layout: article
title: Tricks and trades of making a neural network work.
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---

# Practical tricks to make NNs work
#### Data Set

- Do as much of your work as you can on a small sample of the data ([source 309](https://youtu.be/Th_ckFbc6bI?t=3503))
- Public leaderboard of Kaggle is not a replacement for Validation Set. Jeremy showed how he was ranked in 5th in the private leaderboard of Rossmann Competition, whereas he was not in top 300 in public leaderboard. In another example, The test set of public leaderboard (in Iceberg Satellite image Competition) contained mostly of augmented images. ([source 65](https://youtu.be/sHcLkfRrgoQ?t=2739))
- Look into data. Remove outliers which make sense and there is no other variable to capture those outliers: like in Rossmann Competition, the date&timings for closed stores were not known. There is extra sale before and after close period. So, if you don’t have any data to model the outliers, you need to remove them during training. ([source 37](https://youtu.be/sHcLkfRrgoQ?t=2898))
- Look into training: After training the cats vs dogs we could see that some incorrect classified images were mostly misclassified due to cropping. The solution was data augmentation ([source 23](https://youtu.be/IPBSB1HLNLo?t=1674))
- Data Augmentation: You cannot use all possible types of augmentation. For best results we need to use the right kind of augmentation. ([source 77](https://youtu.be/9C06ZPF8Uuc?t=1408))
- Test Time Augmentation (TTA): Increases accuracy [source 93](https://youtu.be/JNxcznsrRb8?t=3669)
- Rossman Notebook: Without expanding your date-time column using 

add_date_part function of fastai library, you can’t capture any trend/cyclical behavior as a function of time at any of these granularities. We’ll add to every table with a date field.
- Rossman Notebook: many models have problems when missing values are present, so it’s always important to think about how to deal with them. In these cases, we are picking an arbitrary signal value that doesn’t otherwise appear in the data.

#### Learning Rate

- Use learning rate finder and select a learning where convergence of loss is steep. Do not select the biggest possible learning rate. ([source 116](https://youtu.be/JNxcznsrRb8?t=1320))
- When using a pretrained model on some dataset like imagenet, you need to use different learning rates when you are using that model for any new dataset. The initial layers need a smaller learning rate, and the deeper layers need a comparatively larger learning rate. When the new dataset is similar to original dataset (e.g. cats vs dogs is similar to imagenet but iceberg satellite image is not) the weights have a ratio of 10. But when using the imagenet of satellite model the successive weights should have a ratio of 3. ([source 31](https://youtu.be/9C06ZPF8Uuc?t=5858))
- Cosine annealing: This now supported by default in pytorch [0.3.1 36](http://pytorch.org/docs/0.3.1/optim.html#torch.optim.lr_scheduler.CosineAnnealingLR) ([source 65](https://youtu.be/JNxcznsrRb8?t=2291))
- SGD with restarts: 2 setups work very well. ([source 55](https://youtu.be/JNxcznsrRb8?t=3272))
- To tackle Gradient Explosion we use identity matrix for intialization ([source 70](https://youtu.be/sHcLkfRrgoQ?t=7780)) Also allows higher learning rate.

#### Training

- bn_freeze = True: In case you are using a deeper network anything greater than resnet34 (like resnext50), bn_freeze = True should be used when you unfreeze and your new dataset is very similar to the original data used in pretrained model ([source 47](https://youtu.be/9C06ZPF8Uuc?t=1109)) Pytorch is probably the only library which offers this much needed switch.
- On the other hand, when the new dataset is not similar to the original dataset, we start with smaller size 64x64, fit in freezed state, unfreeze and fit. And repeat the process with 128x128 and 256x256 ([source 28](https://youtu.be/9C06ZPF8Uuc?t=5898))
- Kaiming He Initialization: Pytorch has this implemented by default. ([source 55](https://youtu.be/J99NV9Cr75I?t=2844))
- Adam Optimizer has a better version called AdamW

#### Activation functions

- theoretically softmax and logsoftmax are scaled version of each other, but empirically logsoftmax is better
- sigmoid instead of softmax for multi-label classification ([source 62](https://youtu.be/9C06ZPF8Uuc?t=4943))
- applying sigmoid in the end when you know the min and max of output (eg. highest sale and lowest sale is known in the Rossmann data) relieves the neural network and training is faster.([source 13](https://youtu.be/J99NV9Cr75I?t=4101)) This is similar to applying softmax when you know the output should be probability ([source 3](https://youtu.be/9C06ZPF8Uuc?t=4498))
- In hidden state to hidden state transition weight matrices tanh is used ([source 13](https://youtu.be/sHcLkfRrgoQ?t=6237))

#### Architectures

- In nlp, we have to use slightly different betas in Adam Optimizer ([source 77](https://youtu.be/gbceqO8PpBg?t=7175))
- In nlp, we use different dropouts all over the place in a specific LSTM model. These dropouts have to be in a certain ratio. ([source 35](https://youtu.be/gbceqO8PpBg?t=7206))
- In nlp, we use also use gradient clipping. There is no reason why this cannot be used in other models ([source 29](https://youtu.be/gbceqO8PpBg?t=7316))
- All the NLP models probably need a line of code for regularisation ([source 32](https://youtu.be/gbceqO8PpBg?t=7262))
- RNN cell is not used nowadays coz of low learning rate constraint due to gradient explosion. We use GRU
- In sentiment analysis, transfer learning has outperformed state of the art sentiment analysis models [source 23](https://youtu.be/gbceqO8PpBg?t=7738)
- Stride 2 convolution has same effect as Max Pool
- Batch normalisation allows us to design resilient deeper networks and learning rate can be made higher. It is similar to dropout in the sense that it changes the meaning of the layers, which a kind of regularisation technique like dropout. Therefore, like dropout it is not done in test mode. ([source 12](https://youtu.be/H3g26EVADgY?t=5732))
- Batch normalisation works best when done after relU
- Resnet ensures richer input for first layer. 5by5 convolution is used in start, and stride is set to 1. In subsequent layers stride is 2 and 3by3 convolution is used. Padding is important when your activations are smaller like 4by4. ([source 11](https://youtu.be/H3g26EVADgY?t=4972))
- Resnet uses something known as Identity training. It has layer-groups. Each layer-group has a bottleneck layer with stride = 2, which causes reduction in activation size. The rest of the layers in the group just try to predict the error through identity training. This concept is yet to be explored in NLP. ([source 8](https://youtu.be/H3g26EVADgY?t=6913))
- Concatenation of AdaptiveAvg Pooling and AdaptiveMaxPooling is better. ([source 33](https://youtu.be/H3g26EVADgY?t=7611))









# Debugging ML code:
- start with a small model and small data and evolve both together. If you can't overfit a small amount of data you've got a simple bug somewhere. 
    + Start with all zero data first to see what loss you get with the base output distribution, then gradually include more inputs (e.g. try to overfit a single batch) and scale up the net, making sure you beat the previous thing each time.
    + also if zero inputs produces a nice/decaying loss curve, this usually indicates not very clever initialization.
    + initialize parameters with truncated normal or xavier.
    + also try tweak the final layer biases to be close to base distribution
    + for classification, check if the loss started at ln(n_classes)

### sanity checks
- remember to toggle train/eval mode for the net. 
- remember to .zero_grad() (in pytorch) before .backward(). 
- remember not to pass softmaxed outputs to a loss that expects raw logits.
- pytorch `.view()` function reads from the last dimension first and fills the last dimension first too
- when comparing tensors, the results are `ByteTensor`s. ByteTensors have a buffer of `255` after which it is zeroed out. Although this issue seems to be fixed in newer pytorch versions, beware that a `sum()` on ByteTensors is likely to result in wrong answer. First convert them to `float()` or `long()` and then `sum()`


### making the loss go down
If your network isn’t learning (meaning: the loss/accuracy is not converging during training, or you’re not getting results you expect), try these tips:

- Overfit! The first thing to do if your network isn’t learning is to overfit a training point. Accuracy should be essentially 100% or 99.99%, or an error as close to 0. If your neural network can’t overfit a single data point, something is seriously wrong with the architecture, but it may be subtle. If you can overfit one data point but training on a larger set still does not converge, try the following suggestions.
- Lower your learning rate. Your network will learn slower, but it may find its way into a minimum that it couldn’t get into before because its step size was too big. (Intuitively, think of stepping over a ditch on the side of the road, when you actually want to get into the lowest part of the ditch, where your error is the lowest.)
- Raise your learning rate. This will speed up training which helps tighten the feedback loop, meaning you’ll have an inkling sooner whether your network is working. While the network should converge sooner, its results probably won’t be great, and the “convergence” might actually jump around a lot. (With ADAM, we found ~0.001 to be pretty good in many experiences.)
- Decrease (mini-)batch size. Reducing a batch size to 1 can give you more granular feedback related to the weight updates, which you should report with TensorBoard (or some other debugging/visualization tool).
- Remove batch normalization. Along with decreasing batch size to 1, doing this can expose diminishing or exploding gradients. For weeks we had a network that wasn’t converging, and only when we removed batch normalization did we realize that the outputs were all NaN by the second iteration. Batch norm was putting a band-aid on something that needed a tourniquet. It has its place, but only after you know your network is bug-free.
- Increase (mini-)batch size. A larger batch size—heck, the whole training set if you could—reduces variance in gradient updates, making each iteration more accurate. In other words, weight updates will be in the right direction. But! There’s an effective upper bound on its usefulness, as well as physical memory limits. Typically, we find this less useful than the previous two suggestions to reduce batch size to 1 and remove batch norm.
- Check your reshaping. Drastic reshaping (like changing an image’s X,Y dimensions) can destroy spatial locality, making it harder for a network to learn since it must also learn the reshape. (Natural features become fragmented. The fact that natural features appear spatially local is why conv nets are so effective!) Be especially careful if reshaping with multiple images/channels; use numpy.stack() for proper alignment.
- Scrutinize your loss function. If using a complex function, try simplifying it to something like L1 or L2. We’ve found L1 to be less sensitive to outliers, making less drastic adjustments when hitting a noisy batch or training point.
- Scrutinize your visualizations, if applicable. Is your viz library (matplotlib, OpenCV, etc.) adjusting the scale of the values, or clipping them? Consider using a perceptually-uniform color scheme as well.


## Optimize the optimization process of your neural net

1- earlier layers of a deep NN usually converge much faster than later layers meaning that later layers need much more changing before they settle on a converged representation. Therefore, using multiple learning rates (larger for later layers) should help with the total convergence time. 

2- find optimal learning rate by doing a trial epoch. use a low learning rate and increase it exponentially with each batch and record loss for every learning rate. then plot learning rate vs. loss. Optimum learning rate is the highest value of the learning rate where the loss is still decreasing and hasn't plateaued. 

3- As training progresses, model gets closer to the minimum and therefore the learning rate should be decreased to prevent overshooting (oscillatory behavior). Cosine annealing achieves this by having the learning rate decrease with a cosine function usually between [0.1, 1].  $$(LR * cosine(batch_number/(total_batch * num_epochs)) $$.

4-  SGD might get stuck in a local minimum during training, increasing learning rate suddenly might help it hop out of local minima. This is usually done by restarting the annealing cosine scheduling of learning rate. This forms a cycle. We then make this cycle longer as the training evolves by for example starting with restarting every 1 epoch, then every 2 epochs, then every 3, and so on. 

5- know that activation functions have their own characteristics that should be considered when making design decisions. Softmax likes to pick just one thing. Sigmoid wants to know where you are between -1 and 1, and beyond these values won’t care how much you increase. Relu is a club bouncer who won’t let negative numbers through the door.

6- On structured data, you can simply embed categorical variables with embedding vectors like word vectors and use NNs instead of dummy or binary variables and RF-based models.

7- to overcome overfitting:
    - use dropout,
    - train on smaller image sizes, then increasing the size of input and train again. 

8- ensemble your model prediction:
    - test time augmentation (TTA): feed different versions of the test input (e.g. crops or zooms) and pass through the model, use average output as the final output score.



## Use Normalization to ease optimization

Normalization does not reduce the expressive power of the network but normalizes the statistics of the network according to the dataset statistics in order to make the optimization of the network easier. 
    - Batch Norm : computes the mean and variance of each mini-batch and normalizes each feature. the mean and variance will differ for each mini-batch. This dependency causes two main problems:
        + Ideally, we want to use the global mean and variance to normalize the inputs to a layer so may be problematic with small batch_sizes.
        + makes batch normalization difficult to apply to recurrent connections. 
        + an alternative is mean-only batch normalization which only normalizes the mean and skips the variance scaling.
    - Layer Norm: a mini-batch consists of multiple examples with the same number of features. 
        + Batch normalization normalizes the input features across the batch dimension. The key feature of layer normalization is that it normalizes the batch across the features.
        + In batch normalization, the statistics are computed across the batch and are the same for each example in the batch. In contrast, in layer normalization, the statistics are computed across each feature and are independent of other examples.
    - Weight Norm: instead of normalizing the mini-batch, normalizes the weights of the layer. 
        + separates the norm of the weight vector from its direction by rewriting the weights as $$ {w} = \frac{g}{v} v$$ and optimizes both g  and v using SGD.
        + mean and variance are independent of the batch, 
        + weight normalization is often much faster than batch normalization.
        + more computationally efficient in CNNs since number of weights are much smaller than inputs. 
    - Spectral Norm: Normalizes each layer using the eigen values of its weight matrix.


    - Layer normalization (Ba 2016): Does not use batch statistics. Normalize using the statistics collected from all units within a layer of the current sample. Does not work well with ConvNets.

    - Recurrent Batch Normalization (BN) (Cooijmans, 2016; also proposed concurrently by Qianli Liao & Tomaso Poggio, but tested on Recurrent ConvNets, instead of RNN/LSTM): Same as batch normalization. Use different normalization statistics for each time step. You need to store a set of mean and standard deviation for each time step.

    - Batch Normalized Recurrent Neural Networks (Laurent, 2015): batch normalization is only applied between the input and hidden state, but not between hidden states. i.e., normalization is not applied over time.

    - Streaming Normalization (Liao et al. 2016) : it summarizes existing normalizations and overcomes most issues mentioned above. It works well with ConvNets, recurrent learning and online learning (i.e., small mini-batch or one sample at a time):

    - Weight Normalization (Salimans and Kingma 2016): whenever a weight is used, it is divided by its L2 norm first, such that the resulting weight has L2 norm 1. That is, output y=x∗(w/|w|), where x and w denote the input and weight respectively. A scalar scaling factor g is then multiplied to the output y=y∗g. But in my experience g seems not essential for performance (also downstream learnable layers can learn this anyway).

    - Cosine Normalization (Luo et al. 2017): weight normalization is very similar to cosine normalization, where the same L2 normalization is applied to both weight and input: y=(x/|x|)∗(w/|w|). Again, manual or automatic differentiation can compute appropriate gradients of x and w.

## design your experimental and reproducability pipeline:

- use [Sacred](http://sacred.readthedocs.io/en/latest/quickstart.html) to record and reproduce your experiments:
    + install Sacred with `pip install sacred`
    + instantiate the Experiment() class from sacred 
    ```py
     ex = Experiment()
    ```
    + put all your argument inside a config function and decorate it with `@ex.config`
    ```py
    @ex.config
    def my_config():
        args = parse_args()
    ```
    + then define your main function and pass the config you defined to it and decorate it with `ex.automain`
    ```py
    @ex.automain
    def main(args):
        ...
    ```
    + Sacred has a command line interface that lets you interact with and modify your experiment. 


- Make a spreadsheet in style of a tree. Each node in the tree will be a grouping of experiments. Each group will only have a child if there is a question that needs to be answered and the experiment will only answer that question. Each leaf node will then be the result of the experiment with the value of the desired metric and a one-sentence answer to the question of the experiment. 

1- define a metric that you are trying to beat or optimize for (e.g. accuracy)

2- hyperparameters used for the experiment

3- Sub-Project: group of ideas you are exploring

Context: The context may be the specific objective such as beating a baseline, tuning, a diagnostic, and so on.
Setup: The setup is the fixed configuration of the experiment.
Name: The name is the unique identifier, perhaps the filename of the script.
Parameter: The parameter is the thing being varied or looked at in the experiment.
Values: The value is the value or values of the parameter that are being explored in the experiment.
Status: The status is the status of the experiment, such as planned, running, or done.
Skill: The skill is the North Star metric that really matters on the project, like accuracy or error.
Question: The question is the motivating question the experiment seeks to address.
Finding: The finding is the one line summary of the outcome of the experiment, the answer to the question.





## torch tricks:

- `torch.tensor(data, requires_grad = False)` copies the data. to avoid copying the data while changing gradient requirements use `requires_grad_()` or `detach()`. To make a tensor from numpy and avoid copying, use `torch.as_tensor()`. use `.item()` to get the scalar inside a scalar tensor. 






