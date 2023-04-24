---
title: Deep learning models are secretly (almost) linear
layout: post
---

I have been increasingly thinking about NN representations and slowly coming to the conclusion that they are (almost) completely secretly linear inside [^1]. This means that, theoretically, if we can understand their directions, we can very easily exert very powerful control on the internal representations, as well as compose and reason about them in a straightforward way. Finding linear directions for a given representation would allow us to arbitrarily amplify or remove it and interpolate along it as desired. We could also then directly 'mix' it with other representations as desired. Measuring these directions during inference would let us detect the degree of each feature that the network assigns to a given input. For instance, this might let us create internal 'lie detectors' (which there is [some progress](https://arxiv.org/abs/2212.03827) towards) which can tell if the model is telling the truth, or being deceptive.  While nothing is super definitive (and clearly networks are not 100% linear), I think there is a large amount of fairly compelling circumstantial evidence for this position. Namely:

# Evidence for this:

-   All the work [from way back when](https://github.com/tayden/VAE-Latent-Space-Explorer) about [interpolating through VAE/GAN latent space](https://arxiv.org/pdf/2102.12139.pdf). I.e. in the latent space of [a VAE on CelebA](https://arxiv.org/pdf/1807.07543.pdf) there are natural 'directions' for recognizable semantic features like 'wearing sunglasses' and 'long hair' and linear interpolations along these directions produced highly recognizable images

-   Rank 1 or low rank editing techniques [such as ROME](https://arxiv.org/abs/2202.05262) work so well (not perfectly but pretty well). These are effectively just emphasizing a linear direction in the weights.

-   You can [apparently add and mix LoRas](https://www.reddit.com/r/StableDiffusion/comments/11ettz2/anybody_know_if_using_multiple_loras_is_ok/) and it works about how you would expect.

-   You can [merge totally different models](https://stable-diffusion-art.com/models/#Merging_two_models). People working with Stable Diffusion community [literally additively merge model weights](https://www.reddit.com/r/StableDiffusion/comments/11kau9d/what_is_the_point_of_the_endless_model_merges/) with weighted sum and it works!

-   [Logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) works

-   [SVD directions](https://www.lesswrong.com/posts/mkbGjzxD8d8XqKHzA/the-singular-value-decompositions-of-transformer-weight)

-   Linear probing definitely gives you a [fair amount of signal](https://arxiv.org/abs/2002.12327)

-   [Linear mode connectivity](https://arxiv.org/abs/1912.05671) and [git rebasin](https://arxiv.org/abs/2209.04836)

-   [Colin Burns' unsupervised linear probing method](https://arxiv.org/abs/2212.03827) works even for semantic features like 'truth'

-   You can [merge together](https://arxiv.org/abs/2302.04863) [different models](https://arxiv.org/abs/2203.05482) finetuned from the same initialization

-   You can do a [moving average over model checkpoints](https://arxiv.org/abs/1803.05407) and this is better!

-   The fact that linear methods work pretty tolerably well in neuroscience

-   Various naive diff based editing and control techniques work at all

-   [General linear transformations between models and modalities](https://openreview.net/forum?id=8tYRqb05pVn) 

-   You can often remove specific concepts from the model by erasing a [linear subspace](https://arxiv.org/abs/2201.12091)

-   [Task vectors](https://arxiv.org/abs/2212.04089) (and in general things like finetune diffs being composable linearly)

-   [People](https://arxiv.org/pdf/1704.01444.pdf) [keep](https://www.lesswrong.com/posts/cAC4AXiNC5ig6jQnc/understanding-and-controlling-a-maze-solving-policy-network) [finding](https://www.lesswrong.com/posts/nmxzr2zsjNtjaHh7x/actually-othello-gpt-has-a-linear-emergent-world) linear representations inside of neural networks when doing interpretability or just randomly

If this is true, then we should be able to achieve quite a high level of control and understanding of NNs solely by straightforward linear methods and interventions. This would mean that deep networks might end up being pretty understandable and controllable artefacts in the near future. Just at this moment, we just have not yet found the right levers yet (or rather lots of existing work does show this but hasn't really been normalized or applied at scale for alignment). Linear-ish network representations are a best case scenario for both interpretability and control.

For a mechanistic, circuits-level understanding, there is still the [problem of superposition](https://transformer-circuits.pub/2022/toy_model/index.html) of the linear representations. However, if the representations are indeed mostly linear than once superposition is solved there seem to be little other obstacles in front of a complete mechanistic understanding of the network. Moreover, superposition is not even a problem for black-box linear methods for controlling and manipulating features where the optimiser handles the superposition for you.

This hypothesis also gets at a set of intuitions I've slowly been developing. Basically, almost all of alignment thinking assumes that NNs are bad -- 'giant inscrutable matrices' -- and success looks like fighting against the NN. This can either be through minimizing the amount of the system that is NN-based, surrounding the NN with monitoring and various other schemes, or by interpreting their internals and trying to find human-understandable circuits inside. I feel like this approach is misguided and makes the problem way more difficult than it needs to be. Instead we should be working with the NNs. Actual NNs appear to be very far from the maximally bad case and appear to possess a number of very convenient properties - including this seeming linearity -- that we should be exploiting rather than ignoring. Especially if this hypothesis is true, then there is just so much control we can get if we just apply black-boxish methods to the right levers. If there is a prevailing linearity, then this should make a number of interpretability methods much more straightforward as well. Solving superposition might just resolve a large degree of the entire problem of interpretability. We may actually be surprisingly close to success at automated interpretability.

# Why might networks actually be linear-ish?

- 1.)  Natural abstractions hypothesis. Most abstractions are naturally linear and compositional in some sense (why?).

- 2.)  NNs or SGD has strong Occam's razor priors towards simplicity and linear = simple.

- 3.) Linear and compositional representations are very good for generalisation and compression which becomes increasingly important for underfit networks on large and highly varied natural datasets. This is similar in spirit to the way that biology [evolves to be modular](https://www.lesswrong.com/posts/JBFHzfPkXHB2XfDGj/evolution-of-modularity).

- 4.)  Architectural evolution. Strongly nonlinear functions are extremely hard to learn with SGD due to poor conditioning. Linear functions are naturally easier to learn and find with SGD. Our networks use almost-linear nonlinearities such as ReLU/GeLU which strongly encourages formation of nearly-linear representations

- 5.)  Some NTK-like theory. Specifically, as NNs get larger, they move less from their initial condition to the solution, so we can increasingly approximate them with linear taylor expansions. If the default 'representations' are linear and Gaussian due to the initialisation of the network, then perhaps SGD just finds solutions very close to the initialisation which preserve most of the properties.

- 6.)  Our brains can only really perceive linear features and so everything we successfully observe in NNs is linear too, we just miss all the massively nonlinear stuff. This is the anthropic argument and would be the failure case. We just miss all the nonlinear stuff and there lies the danger. Also, if we are applying any implicit selection pressure to the model -- for instance optimising against interpretability tools -- then this might push dangerous behaviour into nonlinear representations to evade our sensors.

[^1]: Of course the actual function the network implements cannot be completely linear otherwise we would just be doing a glorified (and expensive) linear regression.
