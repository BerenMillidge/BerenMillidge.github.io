---
layout: post
title: Linear Attention as Iterated Hopfield Networks
---


*In this short note, we present an equivalence and interpretation of the recurrent form of linear attention as implementing a continually updated hopfield network. Specifically, as the recurrent transformer is performing generation for each token, it simply adds a continuous 'memory' via Hebbian plasticity to a classical continuous hopfield network and then performs a heteroassociative lookup given the query of that token. While mathematically straightforward and following directly from previous results, this interpretation provides a novel lens through which to view mechanistically how linear attention models function, and provides a link between the capacity and expressivity of linear attention models and well-studied problems such as the memory capacity of classical Hopfield Networks which may give insights into the limitations and potential improvements of linear attention transformers.*

Since their introduction in 2017, Transformer models have proven themselves to be an incredibly versatile and powerful sequence modelling approach with [highly predictable](https://arxiv.org/abs/2001.08361) [scaling properties](https://arxiv.org/abs/2203.15556) which have formed the basis of almost all [state of the art](https://papers.nips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) [AI systems](https://arxiv.org/abs/2307.09288) [today](https://arxiv.org/abs/2312.11805) in language modelling. The core architectural component unique to transformers is the [self-attention](https://arxiv.org/abs/1409.0473) sequence mixer block, which enables the model to model correlations and dependencies between different tokens in the sequence in a highly expressive way. While perhaps maximally expressive, and highly efficient to implement on GPU accelerators, self-attention requires $\mathcal{O}(L^2)$ operations and memory for a sequence length $L$ -- such a constraint renders extremely long sequence lengths challenging to work with for transformer models, as well as making autoregressive generation expensive, since it requires a quadratically growing memory overhead for the KV cache for each token in the sequence. These limitations have spurred research into subquadratic architectures which aim to preserve most of the modelling quality of attention while using asymptotically fewer FLOPs and memory. One [influential approach](https://arxiv.org/abs/2006.16236) to this is to linearize the softmax function in the self-attention block, which enables the transformer to be expressed as a recurrent neural network with a matrix-valued state which requires only linear memory and flops for autoregressive generation . While this approach indeed reduces the computational complexity, it appears to come at a substantial cost to performance, at least [until](https://arxiv.org/abs/2312.06635) [recently](https://arxiv.org/abs/2312.00752).

Meanwhile, a different strand of work has sought to understand and [improve transformer attention](https://arxiv.org/abs/2008.02217) through the lens of [associative memories](https://arxiv.org/abs/2008.06996) and Hopfield networks. Here, it has been shown that transformer self-attention [can be cast into a form very similar to that of Modern Hopfield Networks](https://arxiv.org/abs/2008.02217) (see [here](https://www.beren.io/2020-11-02-Walkthrough_Hopfield-Networks-Is-All-You-Need/) for my tutorial on this), a neuroscientifically inspired autoassociative memory model which performs explicit matching and retrieval of stored memories given a query. While an equivalence between softmax attention and Modern Hopfield Networks has been established by prior work, in this note we show that linear attention also has an equivalence to the continuous version of the original classical Hopfield network.

# Transformer Attention

The attention block in transformer models can be expressed as follows: Given an input matrix $x$ of shape $L \times E$ where $L$ is the sequence dimension and $E$ is the embedding dimension, we first compute the 'query', 'key', and 'value' matrices,
$$\begin{align}
    Q = W_Q x \, \, , \, \, K = W_K x \, \, , \, \, V = W_V x
\end{align}$$
We then compute a set of attention scores, 
$$\begin{align}
    y = V \sigma(QK^T)
\end{align}$$
where $\sigma$ is the softmax function $\sigma(x) = \frac{e^{-x}}{\sum e^{-x}}$ (we consider here only a single head and no causal masking for simplicity). Intuitively, what the self-attention block is doing is computing a correlation matrix of similarity scores between the embeddings of each token $QK^T$, then performing a softmax to highlight the largest similarities and suppress the others, and then projecting these scores onto the basis of a set of waiting value vectors in the matrix $V$. Because, in the naive implementation, we materialize the token-token similarity matrix $$QK^T$$, the self attention operation has $\mathcal{O}(L^2)$ complexity.

# Linear Attention

It was noticed by [Katharopoulos et al (2020)](https://arxiv.org/abs/2006.16236) that if we replace the softmax function in self attention with a vector-wise kernel then the computation can both be done in linear complexity and can be interpreted as a recurrent network with a matrix valued state,
$$\begin{align}
    \frac{\text{exp}(-QK^T)V}{\sum \text{exp}(QK^T)} \approx \frac{(\phi(Q)\phi(K^T))V}{\sum \phi(Q)\phi(K^T)} = \frac{\phi(Q)(\phi(K^T)V)}{\sum \phi(Q)\phi(K^T)}
\end{align}$$
where first we replace the exponential in the softmax with two matrix-wise kernels $\exp(-QK^T) \rightarrow -\phi(Q)\phi(K^T)$, then we notice that due to the associativity of multiplication without the exponential function we can rebracket to obtain $(\phi(Q)\phi(K^T))V = \phi(Q)(\phi(K^T)V)$ which transforms the complexity from $\mathcal{O}(L^2d)$ to $\mathcal{O}(Ld^2)$ which can be a significant saving for long sequences (whenever $L > d$). Moreover, while this form can be used for parallel training, the kernelized attention block also admits a recurrent form for autoregressive generation which is linear in both compute and memory overhead. To see this, we write the form of the attention for an arbitrary timestep $$t$$, while also assuming that the kernel function is linear (identity) -- a formulation which has been found to be [surprisingly effective](https://arxiv.org/abs/2210.04243) [in practice](https://arxiv.org/abs/2307.08621),
$$\begin{align}
    y_t = \frac{\sum_{k=0}^t Q_t K_k^T V_k}{\sum_{k=0}^t Q_t K_k} =  \frac{Q_t  \sum_{k=0}^t K_k^T V_k}{ Q_t \sum_{k=0}^t K_k}
\end{align}$$
This can then be expressed as the following recurrence with recurrent state $M_t$ and normalization state $Z_t$
$$\begin{align}
    M_t &= M_{t-1} + K_t V_t^T \\
    Z_t &= Z_{t-1} + K_t \\
    y_t &= \frac{1}{Z_t} M_t q_t
\end{align}$$

This recurrence enables a linear time and memory autoregressive generation since each for each new token only the memory state $H_t$ has to be stored instead of the KV cache of all previous tokens.

# Hopfield Networks

The [classical Hopfield Network](https://www.pnas.org/doi/10.1073/pnas.79.8.2554) was introduced as a model of associative memory intending to model hippocampal function. While the original Hopfield Network dealt with binary vectors, here we present the continuous version with continuous vectors . The Hopfield Network consists of a vector $x$ and a weight matrix storing a set of 'memories' $M$. The Hopfield Network retrieves memories with the update rule
$$\begin{align}
    y = \text{sign}(Mx)
\end{align}$$

Where $\text{sign}$ is the sign function and $b$ is a bias vector. New memories can be added to the memory matrix through the Hebbian weight update,
$$\begin{align}
    M_t = M_{t-1} + \alpha x_t x_t^T
\end{align}$$
where $\alpha$ is a scalar learning rate. Typically the retrieval dynamics are ran for multiple steps (i.e. the $y$ becomes the new $x$ although a single step often suffices for recall. We only consider the single step case here). The retrieval and memory storage dynamics can be expressed as gradient descents on the Hopfield Energy function,
$$\begin{align}
    E = x^T M x
\end{align}$$
The classical Hopfield Network is autoassociative, which means that it retrieves the same memories that it matches a query against. It is also possible to have *heteroassociative* memories which retrieve a different memory than the one matched against. Heteroassociative memories can memorize linked lists of elements -- i.e. if we are at element A, we need to retrieve the next element B. Heteroassociative memories can be expressed simply in the Hopfield Network formalism by splitting the memory matrix $M$ into a 'matching' matrix $K$ and a retrieval matrix $V$ such that $M = KV$. The original autoassociative Hopfield Network uses $M = KK^T$.

While an extremely important model, Hopfield Networks have a strongly limited memory capacity due to interference between similar memories, which can also be expressed in terms of false local minima in the energy function. Recent works have shown this capacity can be [dramatically improved](https://arxiv.org/abs/1606.01164) by using powerful functions such as [higher order polynomials](https://arxiv.org/abs/1702.01929) and [exponentials](https://arxiv.org/abs/2008.06996) to extremize the similarities of memories so only the closest matching memory to the query is returned.

Recently, [I wrote a paper](https://arxiv.org/abs/2202.04557) arguing that this class of associative memory architectures can be expressed in terms of a general framing consisting of two functions -- a *similarity* function which scores the similarity between a 'query' and a set of memory 'keys', and a *separation* function which has the function of extremizing the difference between the similarity scores produced by the similarity function, so as to reduce the interference between similar memory keys. According to this framework, the fundamental operation of all single shot associative memory models can be expressed as,
$$\begin{align}
    y = V \text{sep}(\text{sim}(K \cdot q ))
\end{align}$$
Where $K$ is a matrix of stored 'memories' -- each stored as a vector, $V$ is a second matrix of 'value' or 'retrieval' memories and $q$ is a vector query. For an autoassociative memory system $K = V$, meaning the same memories are matched against and retrieved. Under this framework, the classical hopfield network can be understood as using the dot-product similarity and an identity separation function while the transformer attention block uses the dot-product similarity and a softmax separation function. In this perspective, the key insight in the modern hopfield network literature is to use increasingly more extremizing separation functions to reduce interference -- from linear, to polynomial to exponential and max.

# Equivalence

To see the equivalence between the recurrent linear attention block, we simply have to compare the update and retrieval rules of the classical continuous Hopfield network against the linear attention recurrence. Namely, observe in Equations 5 and 9 that the update rule of the recurrent state in linear attention and the Hopfield weights are identical (the linear attention recurrence assumes an SGD update with a learning rate of $1$). Similarly the recurrent step in the linear attention block (Equation 7) is also identical to a single retrieval step of a continuous classical Hopfield network (Equation 8) up to the normalization term. 

Thus, it is straightforward to give a mechanistic interpretation of what a recurrent step of the linear attention block does in terms of the well-understood Hopfield network model. Specifically,
Given a set of distributed heteroassociative key and value 'memories' stored in the Hopfield memory matrix $M$, and a new pair of key and value memories $k_t, v_t$ and a new query $q_t$.

- First, store the new key and value memories in the memory matrix $M_t = M_{t-1} + k_t v_t^T$.

- Second, perform a classical hopfield memory retrieval given the query $q_t$ and return the retrieved memory.

Despite this overarching similarity, there are some important subtleties which have yet to be addressed. The Hopfield network paradigm assumes that memories and queries are the fundamental fixed objects while all attention variants generate their memories and queries via a learned linear map from the same token embedding. Secondly, most transformer models utilize some kind of masking, such as causal masking for autoregressive generation, which is not considered here.

# Discussion

While mathematically straightforward, we believe this interpretation of linear attention is novel and sheds light both on the mechanistic function of linear attention models as well as their potential limitations. Specifically, this perspectives shows, quite literally, that all the recurrent state is doing in a linear attention block is storing superpositions of key and value memories and then retrieving with the query at each iteration identically to a continuous Hopfield network. This leads immediately to a number of possibilities for improving the model such as whether we should always add new memories with an effective learning rate of $1$ or perhaps this should be data-dependent, whether we always want to add the key and value memories for all tokens since very similar representations might have already been stored, or whether we want any way for the model to be able to 'forget' less useful memories so as to preserve capacity for more relevant information. Recently some of these ideas have already been integrated into improved versions of these models such as explicit gating, varying decay timescales for memories where they appear to have closed much of the gap to standard attention.

A related direction for improvement would be to introduce more powerful separation functions (kernels) to the model which can increase the memory capacity and reduce interference between memories while not increasing the computational complexity. This view of the kernel as separation function is importantly different from the standard view of kernelized attention since it argues that the goal is not simply to [try to approximate](https://arxiv.org/abs/2103.02143) softmax [with kernels](https://arxiv.org/abs/2009.14794), but instead supply any function which is capable of extremizing the differences between the similarity scores of different memories and then normalizing the resulting outputs.

Secondly, this perspective has the potential to offer important quantitative insight into the limitations and capacity of linear attention models by creating a direct link [to the literature](https://www.sciencedirect.com/science/article/abs/pii/0364021388900262#:~:text=A%20mathematical%20framework%20for%20comparing,a%20fraction%20of%20this%20dimension.) studying the [capacity of Hopfield Networks](https://ieeexplore.ieee.org/abstract/document/1057069), [which has been explicitly quantified](https://www.sciencedirect.com/science/article/abs/pii/S0893608099000428) and intensely studied such that there has developed substantial theory around when they can and cannot successfuly store or retrieve memories. We believe that insights from this body of theoretical work may prove useful in enhancing the capabilities of linear attention models.

Conversely, insights from linear attention could be used to improve existing associative memory systems based on Hopfield Networks. One obvious avenue is the normalization term in the recurrence of linear attention parametrized by recurrent state $Z_t$. The function of this term is to return the outputs of the separation function to a useful $\mathcal{O}(1)$ range. Modern Hopfield Networks with softmax automatically include this however others based on polynomials do not and this normalization may improve numerical stability of these models.

