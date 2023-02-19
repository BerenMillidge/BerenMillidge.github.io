---
title: The solution to alignment is many not one
layout: post
---


Here I want to argue against a common implicit assumption I see people making -- that there is, and *must be* one single *solution* to alignment such that when we have this solution alignment is 100% solved, and while we don't have such a solution, we are 100% doomed. 

Instead, successful alignment in practice will probably look like the stacking of many independent and parallel alignment methods and capabilities constraints all at once (security in depth). To make bridges safe, nobody tries to build 'a bridge that it is impossible to destroy' given unlimited resources or energy -- such a construction is impossible from the get go. Instead, we build bridges with various large safety margins and multiple redundancies such that if some subsystems fail the bridge is supported by the redundant systems.  

Similarly, in computer security, where unlike engineering we deal with human adversaries optimizing against us instead of just physics, people do not build provably secure systems, because such systems are extremely expensive at best and impossible at worst [^1]. In practice people create security through a combination of safety margins, multiple overlapping systems of security, and extensive red-teaming (pentesting). 

The reasons for security in depth are mathematically obvious. If we have N alignemnt techniques and to cause x-risk a misaligned AGI needs to break all of them, and each of them are independent, then our probability of failure multiplies rapidly towards 0. Of course the assumptions underlying this analysis may be too rosy. Perhaps the biggest failure mode is the independent assumptions. A sufficiently powerful AGI which is truly misaligned may be able to break them all simultaneously so the correlation between defence mechanisms would go to 1. However, there are fundamental information assymetries that prevent this in practice. The AGI does not know exactly what alignment, safety, and defense techniques we have in place and realistically cannot know this before tripping any alarms if we are careful.

What this means in practice is that alignment solutions can be *imperfect* and can be known to fail in certain circumstances, as long as other approaches can try to constrain the AGI from ever putting it into these circumstances in the first place. Given multiple imperfect methods, we can then stack them to get a system with a much higher degree of overall safety than any method individually. The way reality works is that we get safety through multiple redundant safety systems and large safety margins instead of by building a single perfect, impossible-to-fail system. We cannot write off an alignment proposal just because there is some theoretical possibility of it failing -- this is true for any practical system in the real world, including ones that work in practice! Instead, the important questions when evaluating alignment or safety measures are a.) do they improve our probability of successful alignment or containment enough relative to their cost. b.) how correlated are they with other alignment or safety measures, and c.) To what extent they are synergistic and cover for the weaknesses and vulnerabilities of other approaches.

Broadly, we can split up the methods for AI safety into three facets:

1.) Alignment. Actually trying to get the AGI not to desire to or try to hurt humanity in the first place. This can be through designing good reward functions of human values, preventing deception and mesaoptimization, instilling values into the AGI through whatever method. 

2.) Constraint. Reducing the AGIs capabilities and preventing FOOM. Many issues of alignment only occur at high capabilities levels due to goodheart and so this helps htere are well. Having a system that can optimize arbitrarily against you is bad. Being able to withstand a concrete bounded adversary, however, is just a safety margin.

3.) Containment. Physically, or in software, preventing a misaligned AGI that wants to hurt us from doing so.

Examples of alignment include:
1.) Trying to instill human values through RLHF and supervised learning
2.) Robust grading schemes
3.) AI supervision of the AGIs plans
4.) Designing robust reward functions
5.) Red-teaming
6.) Interpretability to detect deception and mesaopitimization

Examples of constraining AGI optimization power include:
1.) Quantilization
2.) Myopia
3.) Limiting compute/memory
4.) Randomization

Examples of containment include:
1.) Boxing
2.) Red-teaming it in simulation
3.) Tripwires and auto-shutdowns
4.) Continual monitoring.

There are many other proposed methods here and I have just included a few which immediately spring to mind. 

Importantly, for all of these aspects, we can easily stack multiple independent and parallel techniques onto an AGI we wish to make safe. Each of these techniques often has obvious potential failure modes, but also brings with it some probability of success and is not completely corelated with other methods. As such, as we stack more and more of them, our probability of success increases, despite the imperfection of any individual method. 

[^1]: Even if in cases where we can actually prove the security of the system's code, this proof is true only insofar as the system abstractions do not leak. A provably secure codebase can do nothing against physical or social engineering attacks. 
