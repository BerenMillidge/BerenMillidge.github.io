---
title: Why not just stop FOOM?
layout: post
---


AI alignment given FOOM seems exceptionally challenging in general. This is fundamentally because we have no reasonable bounds on the optimization power a post-FOOM agent can apply. Hence, for all we know, such an agent can go arbitrarily off distribution, which destroys our proxies, and also could defeat any method of boxing or other containment method. Intuitively, this seems similar to the difficulty in mathematics of saying things about *all functions in general* vs some set of constrained functions such as continuous functions, infinitely differentiable, lipschitz etc.

AI alignment without FOOM seems much easier. Suppose we constrain the AGI such that it can apply some fixed amount of optimization power and generality which we can conservatively set to a high, but not unlimited bound. This immediately gives us some traction on the problem. We can start to bound how off distribution the AGI might go, and hence the degree to which our proxy objective is goodhearted. Similarly, it would also be possible to bound the amount of optimization power that could be applied to, for instance, break out of a simbox or to mislead our interpretability tools. In general, assuming some kind of bounds appears to be the only way to make the problem tractable at all.

The real challenge, of course, then, is how to enforce this constraint in practice. How can we stop FOOM?

While this is a hard target, I think doing this is much more tractable than alignment of arbitrary post-FOOM agents. Specifically, alignment under arbitrary optimization power requires hitting a potentially very small target. Stopping FOOM requires only requires hitting a large target. Secondly, alignment is fundamentlaly a constructive problem -- we need to construct a solution. Stopping FOOM is a destructive problem -- we just need to limit the self-modification capabilities of the AGI, which seems much easier. Finally, stopping FOOM can be done *now* with the existing pre-FOOM AGI system and doesn't require deep knowledge of the internal representations post-FOOM.

The fundamental element of FOOM that makes it so dangerous is the tight feedback loop of self-modification. If we can break this loop then there is no FOOM. Similarly, FOOM will almost necessarily require the AGI to acquire extremely large amounts of compute and hardware. Thus, if we can penalize these prerequisites strongly enough, we should be able to deter FOOM robustly.

That is, we should apply:

1.) Strong negative rewards to attempting or thinking about self-modification (aided by interpretability tools). Ideally the AGI would learn and develop a strong aversion to self-modification. Moreover, proxy learning will work here since without FOOM, and with careful proxy design, the AGI isn't going to go off distribution enough for our proxies to break down

2.) Strong regularization against modifying its hardware instantiation in general.

3.) Strong regularization against high-impact and accumulating additional resources in general.

There are probably a whole bunch of other possibilities as well. Of course, in reality, there's no reason we shouldn't use *all these methods in parallel*. Every additional method gives us additional multiplicative probability of defense. 

A possible issue is that this functions defacto as an tax on capabilities and hence may be uncompetitive. I disagree strongly with this. There is esesntially *nobody* in the current world who would benefit from FOOM without a strong alignment solution, at least in EV.  Coordinating on preventing FOOM, if it can be done without alignment tax at pre-FOOM capability levels, should thus be something easily coordinated around. Points 1 and 2 seem pretty neutral with respect to current capbilities levels -- for most reasonable tasks we would want an AGI to do, massively self-modify is not one of them. Point 3 would impose an alignment tax insofar as most things we would want an AGI to do would involve *some* impact on the world. However, there is a very large difference in impact between what we would want an aligned AGI to do and converting the lightcone into paperclips. It is possible that we could tune the strength of this regularization to sit in this middle region (ideally being on the conservative side). If we *do* want an aligned AGI to mess with the lightcone, then we should probably work up to it gradually, slowly decreasing the strength of the impact regularization with every increase of our probability of having ultimately solved alignment.

A secondary and highly important problem is figuring out if FOOM is likely to be occur at all. FOOM crucially requires that the returns on cognitive reinvestment are superlinear, or at least not particularly sublinear such that an intelligence explosion can take place that will asymptote far above human level. From what we know of current ML scaling laws, returns on both parameters and compute are extremely strongly sublinear (power law) and hence getting a FOOM simply from throwing more compute will not occur unless we reach a point where the scaling laws break. The real question from here is thus what the scaling laws are for effort spent on *self-modification and improvement* instead of simply increasing compute or data. Narrowing down this question would be highly influential for the alignment strategy and focus we should take, since if FOOM is unlikely, then we will likely be able to solve alignment with iterative experimentation on a spectrum of agent capabilities from sub-human to slightly superhuman. If FOOM is very likely to occur, then we don't have this option and have to either deliberately prevent FOOM, or get alignment perfect on the first try.



