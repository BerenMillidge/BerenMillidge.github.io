---
layout: post
title: "Bayesian Optimization and Importance Sampling"
---
So, I just had a lecture on Monte-carlo sampling methods such as importance sampling, rejection-based sampling, and the beginnings of MCMC, and it was really interesting. Anyway, importance sampling is a pretty simple idea:

We have a distribution P which we don't know, and can't sample from directly, but which, if we are given a point, we can calculate the value of P at that point. We want to compute things with P. Typically integrals and expectations over it, which are obviously intractable analytically, so we need to approximate them and one way to do so is with Monte-carlo.

Monte-carlo is just a fancy name for averaging. We take a number of samples of a distribution, and then average them and that average will converge to the true expectation of the distribution as the number of samples tends to infinity.

But we can't sample from P directly, so we can't do the simple Monte-carlo. Instead we define another distribution Q, which we can sample from. From Q we can sample values which we then plug into P, and obtain our estimates because we are allowed to evaluate P at a specific point.

However, just averaging those would not necessarily provide a good estimate of the expectation of P so we have to weight the values obtained by the ratio of the value of P to the value of Q, so that if Q is high and P is low, then that sample is obviously a bad sample from P, so we should weight it much less in the overall average. The weights are effectively computing the fit of P and Q at a point, and telling us to pay more attention to the times when Q fits P well, and less to the times when it doesn't. This scheme converges asymptotically in the limit of infinite samples.

However this method is critically dependent on getting a good Q. If there are regions of space where P assigns mass while Q doesn't, then convergence is no longer guaranteed, as there are regions of P which will never be sampled from. If Q assigns low mass to regions where P assigns high mass - i.e. if Q is a gaussian and P has a random peak somewhere in the tail, then it will take a huge amount of samples to suitably explore those regions of P to get a good approximation.

So how do we get a good Q? One basic way is simply to pick a wide Q, and eat the large number of samples required to get all the modes of P. One way which I thought up, which is extremely obvious but I struggled to find things exactly like in the literature, is to adjust Q as we go along. Each time we're sampling from Q and evaluating at P, we get information about P. If we have a very expressive parametric representation of Q, we can then use this information to fit Q to P, for instance by minimising a KL divergence, or else fitting Q to a Gaussian Process. We then pick where next to sample in P via bayesian optimisation, probably annealing the trade-off between exploration and expoitation as sampling progresses. We continue the importance-weighting as before and then probably once we've got a relatively fitted Q (for which we can account for all the probability mass of P) then we can sample for a bit from a much better fitted distribution which will speed convergence enormously. This gets us our integral, but it also gets us Q, which is itself a good approximation for P, so in a way we get P too, which can be useful for loads of applications.

This has almost certainly been thought of before, so why isn't it widely used? I'm not sure because I've had only the most cursory look at the literature (it's only half an hour after the lecture finished!), but it could relate to a.) bayesian optimisation in such spaces is difficult, b.) a suitably expressive parametrised representation of Q is not feasible in such high-dimensional spaces, c.) it's generally worse and more difficult than MCMC, which is probably the most likely result, or d.) the computational overhead is not worth it compared to the "brute force" standard Q approach. Nevertheless, I thought it was a cool idea.

Any literature reccomendations in this area would be appreciated.
