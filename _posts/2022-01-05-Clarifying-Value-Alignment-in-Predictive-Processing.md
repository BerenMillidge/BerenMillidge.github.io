---
layout: post
title: Clarifying Value Alignment in Predictive Processing
---

Recently I saw a paper by William Ratoff which argues that the alignment problem is [significantly ameliorated](https://link.springer.com/article/10.1007/s10676-021-09611-0#Sec5) for predictive processing (PP) agents. Their argument can be summarized as follows: 

1.) PP agents learn and act by minimizing prediction errors

2.) Most human behaviour (i.e. the PP agents dataset) historically has arguably been mostly moral and lacks evidence of taking over the world or other clear existential risks

3.) As such a PP agent learning from such a dataset would also not try to take over the world or undertake existentially risky behaviour

4.) Thus, if we build superintelligent AGI with a PP architecture then it will probably be safe.

5.) The first superintelligent AGI is also likely to use a PP architecture, because a.) the brain is thought to use a PP architecture and b.) as such it may be easier to build AGIs with a PP architecture rather than any other

In theory this then provides a simple solution to the value alignment problem -- simply build an AGI with a PP cognitive architecture. Unfortunately, I don't think it's that simple. Specifically, I think that while points 1-4 are largely correct, although with caveats, that 5.) is definitely incorrect as a PP superintelligence as described would be safe, but largely useless and so unlikely to be built at all. The key confusion here is fundamentally about the nature of a PP agent. 

PP agents as described in the paper are solely concerned with minimizing prediction error through both action and perception. As such they do not have specific goals or utility functions (this is crucial) but instead aim to predict their data as accurately as possible. Mathematically this is equivalent to simply maximizing the log likelihood of a dataset under a certain (Gaussian) generative model. As such, the arguments of this paper are actually broader than PP and apply to any agent that perceives and acts simply to maximize the log likelihood of some given dataset of human behaviour

Of course if we simply train a superintelligent AI to maximize the likelihood of the data, where the data are human actions in a variety of situations, this objective is highly likely to be safe, because the resulting superintelligence will just act like an average human. The superintelligence may 'see' ways to take over the world but, as noted in the paper, the agent is very unlikely to actually pursue such actions as they have very low log likelihood -- the average human behaviour does not involve successfully taking over the world (if the dataset is such that this is true then you have other problems). 

But this fundamentally ignores the reason why we would want to build a superintelligence in the first place: to do useful things that humans cannot do. If we just want to create agents that act like the average human, well we're already done. We have plenty of average humans already. All we have done here is rediscover the safety-capability tradeoff in another light.

The real existential risk comes not from agents maximizing the log likelihood on some dataset of human actions but the ability of agents to have arbitrary goals or utility functions where, in the space of all possible utility functions, almost all are not going to result in behaviour 'friendly' to humanity. Ratoff but argues that this 'Humean' approach of separating goals and intellect is incorrect and that instead the PP approach of specifying goals as the minimization of prediction error is the way that humans operate and also the way that we are likely to build AGIs. However, this crucial argument is based on several fundamental misunderstandings:

1.) The PP approach assumed in the paper is not an opponent of the 'Humean' orthogonality thesis but an example -- i.e. PP as Ratoff describes it simply proposes one specific objective function (minimize prediction error of sensations). The orthogonality 'thesis' is simply nothing more than the unarguable mathematical distinction between the objective function and the optimization algorithm and thus cannot be 'incorrect'.

2.) In practice, current ML agents do implement the orthogonality thesis in that reinforcement learning agents are trained to maximize arbitrary reward function, and any agent can be paired with any reward function.

3.) The PP architecture does *not* technically describe an objective function but an optimization procedure and as such also follows the orthogonality thesis: PP agents can be instantiated with arbitrary 'goals'. This is a subtle point which has caused much confusion in the literature. The 'PP cognitive architecture' specifies hierarchical prediction error minimization but does not specify what the prediction errors are *between*. If the prediction errors are between sensory data and predicted sensory data, then PP is equivalent to maximizing the log likelihood of data and is probably safe. However, if the prediction errors are between the current world-state and some *desired world state*, where the *desired world state* is an arbitrary pre-set function, then the PP cognitive architecture produces an agent which can maximize arbitrary utility functions -- thus fulfilling the orthogonality thesis again. 

4.) In practice, this is also how PP architectures for control are implemented, where the *desired world state* is mathematically represented as a prior distribution over world states. However, by the complete class theorem, the prior distribution and utility function are equivalent parametrizations of the notion of goals, and there is a direct mathematical mapping between them.

In short, the paper is correct that 'PP' agents, or really any agents that simply perform imitation learning on a dataset of existing human actions are unlikely to be an existential threat (except insofar as humans already are to themselves), since in this paradigm increases of 'intelligence' simply result in better modelling of human behaviour. However as such agents are generally useless at best and prone to strange failure modes such as the dark-room problem, in practice any real attempt to build a superintelligence will likely instead implement the orthogonality principle with separated goals and intellect as this is currently true of reinforcement learning where agents are assigned arbitrary reward functions, and also for agents with a PP cognitive architecture where 'priors' instead take the role of utilities. As such, the value alignment problem is not ameliorated for PP agents or in general.
