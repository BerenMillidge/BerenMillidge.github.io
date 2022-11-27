---
title: Don't argmax; Distribution match.
layout: post
---

I mentioned this briefly in a previous post, but thought I should expand on it a little. Basically, using argmax objectives, as in AIXI or many RL systems are intrinsically exceptionally bad from an alignment perspective due to the standard and well-known issues of goodhearting, ignoring uncertainty etc. There have been a bunch of ways to try to prevent this such as in designing more robust objectives, satisficing, quantilization etc, and I would like to propose another here: *distribution matching*.

The key idea behind distribution matching is instead of argmaxing a reward function, you use the well-known duality between rewawrd and probability to convert a reward function into a desired probability distribution over states, and then the objective is to minimize the some distributional objective between predicted future world states and the desired distribution. 

There are two ways to think about this. The first intuition is that it provides a natural way to encode something like satisficing behaviour, but more attuned to the specifics of the reward function. Specifically, if the reward function is quite broad, so that there are a number of broadly 'acceptable' solutions, the optimal policy will tend to optimize to ensure that all of them have some serious probability of occurence. Argmax on the other hand will overfit to the precisely optimal solution. This is importance because it is likely with imperfect, noisy, or learnt reward functions that the exact optimum will be unlikely or adversarial in some sense. In the language of high-dimensional probability distributions, distribution matching objectives will tend to optimize for the *typical set* rather than the *mode* of the distribution, and this is likely to be much more robust. 

Another mathematically equivalent way to understand the objective is that it is argmaxing with entropy regularization. That is, the agent wants to argmax a reward function, but also keep the distribution over future states or obserations as high entropy as possible. This also implicitly encourages robustness in solutions as many AGI x-risk scenarios (such as a lightcone full of paperclips) are intrinsically low entropy. The distribution matching objective also has good information-theoretic justifications as well as close relationships to information-maximizing exploration and empowerment which I expand [upon here](https://arxiv.org/pdf/2103.06859.pdf).

By default, as with argmax, the distribution matching objective assumes that you have a known reward function, however the duality between reward and probability makes it trivial to use the techniques of Bayesian inference to both infer reward functions from data as well as to correctly integrate reward uncertainty into a model and be robust to it, unlike argmaxing approaches. 

