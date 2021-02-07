---
layout: post
title: "Thoughts on Recursive Self Improvement"
---
A big factor in AI risk is the possibility of recursive self improvement. An AI is created with some level of intelligence and it uses that intelligence to research AI techniques to make itself more intelligent, and then as it is more intelligent, it becomes better at researching ways to make itself even more intelligent, and so on, until it achieves some kind of intelligence singularity or, at least, becomes vastly more intelligent than any human.

This all seems pretty reasonable, and I think it's largely correct. However I want to examine a few assumptions of the argument. But before that I want to digress a little on the nature of intelligence.

What is intelligence? The truth is we don't know. There are a whole bunch of different and wildly conflicting definitions available. This is probably a result of the fact that our colloquial word "intelligence" really covers a wide range of different phenomena, and that as our understanding of these improves, we will develop many different specialized words for each one. However, at the centre of all the definitions, there seems to be a nugget of commonality.

Let's make a first stab at a definition:

Intelligence, broadly, is the ability of a system to respond to its environment in "good" ways.

This definition a.) is leaves open what "good" is, and b.) is basically synonymous with effectiveness, so let's refine this a little.

The first problem in some sense isn't really a problem at all.  The difference between ability and values has long been known - in philosophy it is known as the is-ought gap.  This is stated in more modern terms as the "orthogonality" thesis, which is that the intelligence and the values of an agent are theoretically orthogonal.

This thesis only holds, however, in the absence of any sort of selection pressure. If an agent must survive in some kind of evolutionary environment, then, at minimum, the agent must possess to some degree a set of values known as the Omohundro drives. These drives are prerequisites for achieving the vast majority of goals and are thus likely to appear as auxiliary instrumental goals in any system, and in almost all systems that exist in or have originated from some selection regime. They include such desires as survival, and self-improvement - to achieve a goal you must first survive, and b.) for most goals you are better off being more capable than less in a general sense.

The second also isn't as problematic as it sounds.  Many aspects of intelligence, such as reasoning, have been solved in a normative sense. Assuming you want a method of reasoning that obeys some fairly common-sense constraints, then Bayesian updating is normatively the correct way to reason.




An intelligent agent, then, strives to achieves its arbitrary goals. Its intelligence, broadly, is defined by how effective it is at achieving them. Intelligence thus includes such cognitive tasks as planning and reasoning under uncertainty, creativity in response to new circumstances, and rapid improvisational capacity. Whether this intelligence is in some sense unitary - as in there is a "master algorithm" or whether it is the result of multiple specialised subsystems glued together is difficult to say at this point, and I will remain agnostic.

Regardless of the precise architecture, it seems likely that the intelligence can be split into two components -  an algorithmic component and a hardware component. An AI can become more intelligent either by improving its algorithms or by integrating additional computing power. The recursive self improvement largely assumes that a major component of its intelligence potential lies in algorithmic improvements as the acquisition of hardware is nowhere near as dependent on raw intellect as algorithmic improvements are.

But this is not necessarily the case. It is easy to imagine a world in which the algorithmic implementation of intelligence is "solved".  This is not as far fetched as might seem. On an abstract sense, intelligence defined as reasoning is already solved. Bayesian updating is provably the optimal way to combine information and update beliefs.

Furthermore, optimal agents can also be defined. AIXI is such an example.The trouble with both bayesian updating and agents like AIXI is that over large continuous statespaces such as the world as we know it,  is that they are mathematically intractable.

For instance,  consider bayesian reasoning in a real world case. Here both the prior and likelihood must be continuous probability distributions - i.e. they assign some probability density to every possible division of space. As there are an infinite amount of these divisions, we must, if we are to implement bayesian inference optimally, compute over an infinite amount of points, which is obviously not possible.

Obviously, however, we can do bayesian inference. So we must approximate somehow to remove the infinities. A simple solution would be to discretise the continuous distribution into a discrete one, where complete enumeration of all points is theoretically possible, and thus optimal inference can be carried out. This discretisation process must always lose some information, however, and in practice if done at any reasonable fidelity yields a system with too many discrete points to reasonably consider in any case.

The solution, then, is instead to restrict the possible prior and likelihood functions from any arbitrary distribution to the subset consisting of well defined parametrised distributions, such as Gaussians. These can be modelled with only a few parameters and are often mathematically tractable,but a huge amount of information is lost.

The choice of approximation to make is implicitly a model. Better models then are ones which better reflect the true distrbutions in the world, and so can be more easily used for optimal inference. A significant component of intelligence seems to be coming up with good models of the world. A crucial part of "understanding" a concept seems to be creating a model for it.

However, model choice is itself an inference problem: given a set of percepts, what is the "best" model that accounts for them. However here the same problems arise as in object level reasoning. One way to find the best model is to score all models according to some functoin - usually a likelihood function: the likelihood of the percepts being generated by the model.  The space of all possible models is of course infinite too, so we need a way to approximate it somehow. We need a model of models.

This infinite regress can keep going indefinitely.  But perhaps the higher level metaspaces are esaier to approximate well than the lower levels. If this is the case then the regress can be bounded and inference "down the chain" can begin.  From there we can thus derive a near optimal algorithm for model building - for intelligence. The capacity of the algorithm and its models would only be limited by the amount of computing power available, and thus the level of approximation. If a provably optimal model building algorithm could be developed, then intelligence would be solved.  The only way to become more intelligent would be adding additional computational power to make tractable more detailed and accurate models.

Along with reasoning, the other main core of intelligence must be search.  Many problems which are seen as the cornerstone of intelligence can be reconceptualised as search problems.  For instance consider the problem of proving mathematical theorems. A proof is, at root, a series of applications of mathematical rules to the mathematical axioms. We could then consider the incomprehensibly vast space of all possible combinations of rule applications up to some number N as our search space. The task, then, is finding within that space those sequences that correspond to valid proofs.

A similar method can be described for all intelligent and creative tasks. Consider the task of writing the next great American novel. We can consider this task as searching through the space of all sequences of words up to, say, 500,000 words in length to find one sequence which would result in a classic novel.

Incidentally, finding better AI algorithms also counts as a search problem - from the space of all possible algorithms, find the ones better than the current one. The same also applies to search algorithms, leading to the possibility of recursive search improvement. If we consider intelligence largely isomorphic to search, then recursive search and recursive intelligence improvement are the same thing.

Here modelling is also vital. Doing intelligent and creative work, however, does not feel like a search problem, at least phenomenologically. The reason for this is that our "intelligence algorithm" has two tools is uses to make the search space tractable. The first is abstraction. This allows us to collapse many parts of the search space into one, and to treat regions which behave similarly as instances of a single type so theydo not need to be expored individually. The second is model building. If we have a model of the statespace, we can eliminate vast regions from contention immediately, and focus only on those which the model says are likely to be useful.  The phenomenological experience of intelligence is not so much searching a vast basic statespace, but rather building an ever more refined model to single out just the region that contains the answer. Through the combination of abstraction and model building we manage to perform an incredible feat of search with barely any interaction at all with the underlying search space.

Reasoning, and search, then seem to be the two high level abilities that are required for a general intelligent system. For reasoning we know the optimal algorithm - Bayes' rule. The key place for algorithmic improvements comes from algorithms that make and update better models, and are able to change the models' structure in a principled way to represent the data best. This aspect is largely missing in current AI systems where the high level network architecture is set directly by the human designer and cannot change itself autonomously. A system which is able to optimise its architecture efficiently would be a significant breakthrough.

For search, a similar problem arises. We know of very good algorithms such as A* and other heuristic algorithms. But they still struggle with very large state spaces. For an AI to be effective in this area, it must be able to synthesise all its knowledge to build models that can eliminate vast swathes of the state space a priori. This is an unsolved problem too.

There is thus significant potential for algorithmic improvements in current AI systems. However, for recursive self improvement to be effective, these improvements must be exponential, or at least linear. If they are sublinear then the recursion will eventually ground to a halt. It is difficult at this stage to forsee whether this will be the case. Most likely there will be an initial burst followed by diminishing returns.

Hardware improvements will follow a similar pattern. Exponential improvements are unlikely. Linear is probably the best to hope for, but the likely result will be a linear phase until it drops to sublinear as the overheads of communicatoin and coordination impose ever larger burdens on the system




Ultimately, any account of recursive self improvement must reach the phase of diminishing returns. An optimal algorithm must exist, and results from computational complexity theory will ultimately bound the performance of any algorithm. Once this phase is reached, the only further incremenets in intelligene must come from additional hardware. And here too gains must eventually hit diminishing returns. This means that intelligence is, in some sense, finite,  although the bound is undoubtedly incomprehensibly higher than anything we can understand today.

What does all of this mean for the theory of recursive self improvement, then? Despite a lot of verbiage, I must conclude that not a lot has changed.  There is still too much uncertainty in the dynamics of intelligence enhancement to figure out whether an exponential increase (for a time) is likely. Intuitively, it seems so, as the AIs cognitive architecture is so much more modifiable than ours, but this may not be the case. On the other hand, as with sufficiently general algorithms, improvements in effective intelligence can be obtained smiply by adding more hardware, itmight suggest thatany AIs first task will be to simply acquire more processing power, no matter it's level of intelligence. This implies that the level of intelligence necessary for an exponential explosion is not necessarily bounded by the actual level of inteligence of the AI, but rather the ease of obtaining hardware.  This means that AI boxing techniques, while certainly not infallible, might nevertheless significantly slow down the rate of self-improvement.





