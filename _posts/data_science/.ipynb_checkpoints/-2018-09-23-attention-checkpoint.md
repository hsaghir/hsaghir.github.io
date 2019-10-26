---
layout: article
title: Attention
comments: true
categories: data_science
image:
  teaser: jupyter-main-logo.svg
---


### Attention Mechanism

http://akosiorek.github.io/ml/2017/10/14/visual-attention.html

- Attention serves the need for translating long sequences to long sequences since a single last vector of encoder can't remember everything about the sequence. However, attention mechanism is simply giving the network access to its internal memory, which is the hidden state of the encoder. In this interpretation, instead of choosing what to “attend” to, the network chooses what to retrieve from memory.
- how do models decide which positions in input seq/memory to focus their attention on? They actually use a combination of two different methods: content-based attention and location-based attention. 
    + Content-based attention allows model to search through their memory and focus on places that match what they’re looking for. 
        * In this case, attention required a query and a similarity function. It has to compare the similarity of the query and every item in memory (e.g. using a dot-product).
    + while location-based attention allows relative movement in memory, enabling the models to loop.
- In seq2seq models, the attention distribution is usually generated with content-based attention. The attending RNN generates a query describing what it wants to focus on. Each item is dot-producted with the query to produce a score, describing how well it matches the query. The scores are fed into a softmax to create the attention distribution.
    + First we need a function to compare target and source hidden states. e.g.
        * $$h_t . h_history$$, 
        * $$h_t . W_att . h_history$$, 
        * $$v_att . tanh(W_att[h_t; h_history])$$
    + Second, we need to convert the output of comparison into relevence weights. Hard attention is a one-hot representation of relevent word to each word but since that's discrete and non-differentiable, people usually use soft attention by applying a softmax to the outputs of the comparison function to get alignment weights. 
    + Then a context vector is built using the weighted averages of the input sequence hidden vectors with above-mentioned alignment weights
    + Conditioned on the concatenation of the context vector and decoder input, the decoder now can compute its next state. 

- Fast weights [Hinton] act like a kind of attention to the recent past but with the strength of the attention being determined by the scalar product between the current hidden vector and the earlier hidden vector. The input x(t) is the context used to compare to previously stored values h.
    + In a fast associative memory there is no need to decide where or when to write to memory and where or when to read from memory. The fast memory is updated all the time and the writes are all superimposed on the same fast changing component of the strength of each synapse
    + Every time the input changes there is a transition to a new hidden state which is determined by a combination of three sources of information:
        * The new input via the slow input-to-hidden weights, C,
        * The previous hidden state via the slow transition weights, W_h,
        * And the recent history of hidden state vectors via the fast weights, A.
    + The effect of the first two sources of information on the new hidden state can be computed once every time point, while the effect of fast weights involves a brief iterative process at each time step. 
    +  Assuming that the fast weights decay exponentially, the effect of the fast weights on the hidden vector during the iterative process is to provide an additional input.
        * This additional input is proportional to the sum over all recent hidden activity vectors weighted by the decay rate raised to the power of how long ago that hidden vector occurred.
    + The update rule for the fast memory weight matrix, A, is simply to multiply the current fast weights by a decay rate, λ, and add a proportion (learning rate) of the outer product of the hidden state vector, h(t).
    + The next vector of hidden activities, h(t + 1), is computed in two steps. The “preliminary” vector h0(t + 1) is computed like a normal LSTM, . The preliminary vector is then used to initiate an “inner loop” iterative process which runs for S steps and progressively changes the hidden state into h(t + 1) = h_S(t + 1), i.e. $$h_{s+1}(t + 1) = f(h0(t + 1) + A(t)h_s(t + 1))$$.

- attention comes at a cost. We need to calculate an attention value for each combination of input and output word. If you have a 50-word input sequence and generate a 50-word output sequence that would be 2500 attention values.
    + This can be solved by attending to both the input and output, the way that DRAW does. 
    + An alternative approach to attention is to use RL to predict an approximate location to focus to. That sounds a lot more like human attention, and that’s what’s done in Recurrent Models of Visual Attention.


#### Transformer network (Attention is all you need) 

- Transformer network replaces the sequential processing part of seq2seq networks (i.e. LSTM or CNN) with a **multihead self-attention** to solve the problem of relating two symbols from input/output sequences to a constant O(1) number of operations. It the  consists of two main parts:
    + Multihead attention
        * In terms of encoder-decoder, the query (Q) is usually the hidden state of the decoder. Values (V) are encoder hidden states that need to be attended to, and the keys (K) are learned parameters of the attention matrix that produce a distribution representing how much attention each value gets. Output is calculated as a wighted sum of values.
        * Multihead attention simply projects the Q, K, and V into a smaller embedding space $$d_v$$ using h=8 different linear mappings and applies the attention function there in parallel. Then concatenate the h=8 embedded attentions and map them back to the original dimension.
        * In self-attention, queries,keys and values that comes form same place i.e. the output of previous layer in encoder. In decoder, self-attention enables each position to attend to all previous positions in the decoder.
            - self-attention connects all positions with O(1) number of sequentially executed operations. 
            - The shorter the path between any combination of positions in the input and output sequences, the easier to learn long-range dependencies.
```python
def attention(Q, K, V):
    num = np.dot(Q, K.T)
    denum = np.sqrt(K.shape[0])
    return np.dot(softmax(num / denum), V)
```
    + Feed forward network
        * In RNN (LSTM), the notion of time step is encoded in the sequence as inputs/outputs flow one at a time. In FNN, the positional encoding must be preserved to represent the time in some way to preserve the positional encoding.
            - One way is to embed the absolute position of input elements (as in ConvS2S).
            - In case of the Transformer authors propose to encode time as sine wave, as an added extra input. Such signal is added to inputs and outputs to represent time passing.


```python
Stage1_out = Embedding512 + TokenPositionEncoding512
Stage2_out = layer_normalization(multihead_attention(Stage1_out) + Stage1_out)
Stage3_out = layer_normalization(FFN(Stage2_out) + Stage2_out)

out_enc = Stage3_out
```

```python
Stage1_out = OutputEmbedding512 + TokenPositionEncoding512

Stage2_Mask = masked_multihead_attention(Stage1_out)
Stage2_Norm1 = layer_normalization(Stage2_Mask) + Stage1_out
Stage2_Multi = multihead_attention(Stage2_Norm1 + out_enc) +  Stage2_Norm1
Stage2_Norm2 = layer_normalization(Stage2_Multi) + Stage2_Multi

Stage3_FNN = FNN(Stage2_Norm2)
Stage3_Norm = layer_normalization(Stage3_FNN) + Stage2_Norm2

out_dec = Stage3_Norm
```