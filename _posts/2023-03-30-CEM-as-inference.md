---
title: CEM as inference
layout: post
---

**Author's note:** *I originally wrote this draft in mid 2020 as maths for a paper I never got around to writing. I think it may be somewhat valuable but primarily archiving for historical reasons.*

In this post, we present a quick and novel derivation of the cross-entropy method (CEM) algorithm derived explicitly as an inference algorithm. We first show how standard CEMcan be derived from a maximum likelihood objective using a standard mathematical procedure to relate inference and optimization problems. Secondly, we go beyond simple ML and extend CEM with a Gaussian prior which leads to an analytically tractable maximum-a-posteriori inference scheme.  This scheme can similarly be treated as variational inferencewith a Gaussian variational distribution. 

The advantage of these manipulations is that firstly, by locating CEM within the inference framework, we can understand more deeply and precisely its mathematical form and the assumptions it makes. Moreover, we can use the standard machinery of Bayesian inference to design principled extensions to this algorithm to handle different cases or prior knowledge.

Finally, we demonstrate one case of this where we assume we have a Gaussian prior which we want to take advantage of in the inference scheme. One potential use of this prior is to act as a regularizer and prevent the CEM algorithm from taking too large a step. This approach has been utilized widely, especially in reinforcement learning, where it goes by the name of 'trust-regions'. Implementationally, it is extremely straightforward
and simply sets the Gaussian prior to be the parameters obtained in the last iteration step.

First, we derive CEM directly from first principles as an iterative maximum likelihood scheme. We begin with the objective function,

$$\begin{align}
    \underset{\{\mu,\sigma \}}{argmax} E_{q(x; \mu,\sigma)} f(x)
\end{align}$$

Where $$f(x)$$ is our quality function -- here we assume that $$f(x) = 1 \, \text{iff} x \geq T \, else \, 0$$. I.e. it is a threshold function with threshold $$T$$. We assume that our search distribution $$q(x; \mu, \sigma)$$ is gaussian (and throughout univariate).
Anyhow, we optimize this under our search distribution straightforward by maximum likelihood. Our strategy is to differentiate the objective (1) and set the derivative to 0. First, we note that as the first step we can apply the log-derivative trick to rewrite the derivative as:

$$\begin{align}
    \frac{d}{d\mu} E_{q(x; \mu,\sigma)} f(x) &= \frac{d}{d\mu} \int q(x;\mu,\sigma) f(x) \\
    &= \int f(x) \frac{d}{d\mu}[q(x; \mu, \sigma)] \\
    &= \int f(x) q(x;\mu,\sigma) \frac{d}{d\mu}[\ln q(x; \mu, \sigma)] \\
    &= E_{q(x; \mu,\sigma)}[f(x)\frac{d \ln q(x;\mu,\sigma)}{d\mu}]
\end{align}$$
Ths penultimate trick is the log derivative trick and is quite clever. It arises directly from the fact that $$\frac{d}{dx} \ln(f(x)) = \frac{1}{f(x)} \frac{df(x)}{dx}$$ and thus $$\frac{df(x)}{dx} = f(x) \frac{d}{dx} \ln(f(x))$$. Given that we have the derivative in equation 5 to optimize, we approximate the expectation under $$E_q$$ with monte-carlo sampling. Moreover, all samples under the threshold are eliminated by the quality function. Our objective thus reduces to
$$\begin{align}
    E_{q(x; \mu,\sigma)}[f(x)\frac{d \ln q(x;\mu,\sigma)}{d\mu}] &\approx \sum_{x_i \sim q(x) \geq T} \frac{d \ln q(x;\mu,\sigma)}{d\mu} \\
    &= \sum_{x_i \sim q(x) \geq T} \frac{d \ln \mathcal{N}(x; \mu, \sigma)}{d\mu}
\end{align}$$
Where we have used the fact that $$q(x)$$ is gaussian. Next we can just substitute in the definition of the gaussian density and analytically solve first for the mean and then the variance. We obtain:
$$\begin{align}
\sum_{x_i \sim q(x) \geq T} \frac{d \ln \mathcal{N}(x; \mu, \sigma)}{d\mu} &= \sum_{x_i \sim q(x) \geq T} \frac{d}{d\mu}[\ln 2 \pi \sigma + \frac{(x_i - \mu)^2}{2\sigma^2}] = 0 \\
&= \sum_{x_i \sim q(x) \geq T} \frac{(x_i - \mu)}{\sigma^2}  = 0 \\
&= \sum_{x_i \sim q(x) \geq T} x_i - N \mu = 0 \\
\implies \mu^* = \frac{1}{N} \sum_{x_i \sim q(x) \geq T} x_i
\end{align}$$
Which is of course the sample mean. We can apply a similar approach to the variance.
$$\begin{align}
    \sum_{x_i \sim q(x) \geq T} \frac{d \ln \mathcal{N}(x; \mu, \sigma)}{d\sigma} &= \sum_{x_i \sim q(x) \geq T} \frac{d}{d\sigma}[\ln 2 \pi \sigma + \frac{(x_i - \mu)^2}{2\sigma^2}] = 0 \\
    &= \frac{1}{\sigma} + -\sum_{x_i \sim q(x) \geq T} \frac{(x_i - \mu)^2}{\sigma^3} = 0 \\
    &= N\sigma^2 - \sum_{x_i \sim q(x) \geq T} (x_i - \mu)^2 = 0 \\
    \implies {\sigma^2}^* = \frac{1}{N} \sum_{x_i \sim q(x) \geq T} (x_i - \mu)^2
\end{align}$$
Which is of course the sample variance. When applied iteratively these updates define the CEM update scheme completely.

**Derivation of CEM-VI**

Now supposing that instead of directly solving the maximum likelihood problem, we introduced a prior and solved a variational inference problem based on the variational free energy. Our objective functional thus becomes the VFE, which we can see decomposes (under a transformation of the quality function $$f(x)_{ML} = \ln f(x)_{VI}$$ which changes nothing since the log is a monotonically decreasing function) into the original CEM objective plus an additional KL divergence term between the prior and the search distribution. Thus this objective impels the optimisation of the CEM objective while also keeping the search distribution as close to the prior search distribution as possible. 
$$\begin{align}
    VFE &= KL[q(x;\mu_q,\sigma_q) \Vert p(x;\mu_p,\sigma_p)f(x)] \\
    &= -\underbrace{E_{q(x;\mu_q,\sigma_q)}[\ln f(x)]}_{\text{CEM Objective}} + \underbrace{KL[q(x; \mu_q, \sigma_q) \Vert p(x; \mu_p, \sigma_p)]}_{\text{Prior Divergence}}
\end{align}$$

Next we show how we can solve this new objective for the optimal mean and variance analytically assuming that the prior is also gaussian. We use the same strategy of just taking the derivative and setting it to 0.
$$\begin{align}
    \frac{d}{d\mu}[VFE] &= -\frac{d}{d\mu}E_{q(x;\mu_q,\sigma_q)}[\ln f(x)] + \frac{d}{d\mu}KL[q(x; \mu,\sigma) \Vert p(x; \mu_p, \sigma_p)] \\
    &= -E_{q(x;\mu_q,\sigma_q)}[\ln f(x) \frac{d}{d\mu}\ln q(x; \mu_q,\sigma_q)] + \frac{d}{d\mu}KL[q(x) \Vert p(x; \mu_p, \sigma_p)] \\
    &= - \sum_{x_i \sim q(x) \geq T} \frac{(x_i - \mu)}{2\sigma_q^2} + \frac{(\mu_q - \mu_p)}{2\sigma^2_p} = 0 \\ 
    &= -\sigma_p^2 [ \sum_{x_i \sim q(x) \geq T} x_i - N\mu_q] + \sigma_q^2 (\mu_q - \mu_p) = 0 \\
    &= -\hat{\mu} + \mu_q + \frac{\sigma_q^2}{N\sigma_p^2}(\mu_q - \mu_p) = 0 \\
    &\implies \mu_q^* = \frac{\hat{\mu} + \alpha \mu_p}{1 + \alpha}
\end{align}$$
Where $$\alpha = \frac{\sigma^2_q}{N\sigma_p^2}$$ and $$\hat{\mu}$$ is the sample variance. We thus see that the form for the optimal $$\mu$$ including the prior takes the form of a smoothed average of the maximum likelihood estimate $$\hat{\mu}$$ and the prior mean $$\mu_p$$. Smoothing schemes like this have been proposed heuristically in the literature. What this bayesian approach does is to provide the optimal smoothing coefficient $$\alpha$$ as the ratio of the posterior and prior variances, weighted by the number of data points N.

Similarly, we can also derive the optimal variance of the new posterior analytically. 
$$\begin{align}
    \frac{d}{d\sigma}[VFE] &= -\frac{d}{d\sigma}E_{q(x;\mu_q,\sigma_q)}[\ln f(x)] + \frac{d}{d\sigma}KL[q(x; \mu_q, \sigma_q) \Vert p(x; \mu_p, \sigma_p)] \\
    &= -E_{q(x;\mu_q,\sigma_q)}[\ln f(x) \frac{d}{d\sigma}\ln q(x; \mu_q,\sigma_q)] + \frac{d}{d\sigma}KL[q(x) \Vert p(x; \mu_p, \sigma_p)] \\
    &= \frac{d}{d\sigma}[- \sum_{x_i \sim q(x) \geq T} \frac{(x_i - \mu)^2}{2\sigma_1^2} - \ln 2 \pi \sigma_q] + \frac{d}{d\sigma}[\ln(\frac{\sigma_p}{\sigma_q} + \frac{\sigma_q^2}{2\sigma_p} + \frac{(\mu_q - \mu_p)^2}{2\sigma_p^2}] = 0 \\
    &= \sum_{x_i \sim q(x) \geq T} \frac{(x_i - \mu_q)^2}{\sigma_q^3} - \frac{1}{\sigma_q} - \frac{1}{\sigma_q} + \frac{\sigma_q}{\sigma_p^2} = 0 \\
    &= \frac{1}{N} \sum_{x_i \sim q(x) \geq T} (x_i - \mu_q)^2  - 2\sigma_q^2 + \frac{\sigma_q^4}{\sigma_p^2} = 0 \\
    &= \frac{\sigma_q^2}{\sigma_p} - \sqrt{2}\sigma_q + \hat{\sigma} = 0
\end{align}$$
This latter expression is a quadratic equation in $$\sigma_q$$ which can be solved straightforwardly to obtain the optimal final posterior variance. While perhaps not as elegant as the other derivations, it does make sense as the quadratic equation contains terms involving all of $$\sigma_p, \sigma_q$$ and the ML CEM result the sample variance $$\hat{\sigma}$$. 

To convert this method into an EM-algorithm all we have to do is iterate equations (23) and (29) to obtain the optimal $$\mu^*$$ and $$\sigma^*$$ in the E-step. The M-step is trivial and simply involves setting the parameters of the old prior to be that of the newly derived posterior. 

Overall, we thus have a direct link between:

a.) Maximum likelihood and CEM,

b.) the EDA objective and the CEM objective,

c.) understand the relationship between variational inference VFE derived objectives and the ML CEM objective, and how it effectively is the same as having a prior in the bayesian framework and 

d.) derived analytical update rules for the optimal means and variances under the VFE objective and thus constructed a variational EM CEM algorithm.
