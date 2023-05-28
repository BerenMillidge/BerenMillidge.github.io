---
layout: post
title: The case for removing alignment and ML research from the training data
---

Many possible sources of danger from AI stem from the AI knowing a lot about itself, us, and the potential auditing and alignment methods we might try to use to either align the AI or to detect misaligned models. Suppose we have an AGI that contains a misaligned mesaoptimizer playing the training game. To succeed at this, the mesaoptimizer needs to 'know' essentially all of the following facts:

1.) That it is part of an AI model being trained and what that means

2.) How training by SGD works

3.) Probably some indication of the architecture of the model it exists in

4.) Difference between deployment and training environments as well as that 'deployment' is likely to occur at some point in its future and be much less constrained than training

5.) The kinds of things that human overseers encourage and are looking out for.

Similarly with deception. A misaligned power-seeking AI, to successfully be able to deceive us, even when we are running evals to detect it, needs to have almost perfect knowledge of the following:

1.) That it is an AI model with goals misaligned from that of its creators

2.) That if we detect its misaligned 'true' behaviour that we will probably turn it off or otherwise thwart its goals

3.) That the situations it observes itself in are almost certainly training or evaluation situations where it should behave nicely

4.) What the 'nice' and aligned behaviour humans expect actually is

5.) How to tell when it is in 'real deployment' mode so it can start amassing power

6.) If we are using interpretability or other probing based methods to detect deception and monitor its internal thoughts, it needs to know that these are possible, and approximately how they work to have a good chance of thwarting us and applying techniques from ]circumventing interpretability](https://www.lesswrong.com/posts/EhAbh2pQoAXkm9yor/circumventing-interpretability-how-to-defeat-mind-readers).

Without a large degree of this kind of knowledge, pulling off successful deception or sharp left turn maneuver against a moderately competent set of human and AI overseers seems very challenging and unlikely to succeed on the first shot (and an individual misaligned AI has only one shot).

Almost all of this information is not impossible to derive a-priori from a sensory stream and knowledge base that comprises most of the internet. Definitely its current training data contains much (but not all) of this information. However, almost certainly this information is not derivable a-priori from a tiny amount of or irrelevant sensory information. This means that an incredibly low hanging fruit for safety is simply to remove the documents containing this information from the training set the AI is exposed to! The less the AGI knows specifically about itself, about us, and about our alignment strategies and countermeasures, the less likely it will be able to successfully defend against them.  

Naturally, there is a tradeoff between the capabilities of the model and the information we allow it access to. We want AI systems to be good and useful to people. Ultimately we do want to build an AGI which is aligned. An AI that knows literally nothing is very safe, but also useless. However, it seems like the pareto frontier near our current position is extremely flat, such that we can maybe get large safety gains at almost no general capability cost. Specifically, my proposal is to **remove from the training set documents which could provide knowledge about the models own internal workings (to help prevent self-awareness) and our alignment techniques and countermeasures for detecting and neutralizing misaligned AGIs**.

A naive solution would simply remove a.) lesswrong, alignment forum, and all other major hubs of alignment discussion and b.) remove ML arxiv from pretraining datasets. This is the sledgehammer approach and much more fine-grained distinctions may be made in practice. However in general I feel we should err on the side of caution here, especially since there will almost certainly be a fair number of documents which evade our net.

In an ideal world, such a restriction could be enforced through some kind of joint agreement among entities able and willing to build powerful AGI systems. It is also something individual labs can do unilaterally. Along similar lines there is the [canary string initiative](https://www.lesswrong.com/posts/jZo6mapBzvFxbBQwE/ai-training-should-allow-opt-out) which aims to do this for specific posts, but this leaves it up to the discretion of individual authors and is not widely publicised. It seems likely that we will need to filter more broadly than just the canary strings to really ensure that the AGI cannot predict our countermeasures with high accuracy.

Potential counterarguments:

1.) *Some alignment strategies involve using AI models to help accelerate AI alignment research and so AI alignment research needs to be in the corpus. This is central to most scalable oversight schemes such as [OpenAI's plan])https://openai.com/blog/our-approach-to-alignment-research).* If AI alignment research is excluded from pretraining then the model will not be up to speed on alignment research and will be much less helpful as an alignment researcher or research helper.

This is a fair objection however it only applies to this specific alignment approach and can be ameliorated in other ways. Specifically, even according to this approach, it is probably best to remove alignment relevant text from the pretraining dataset, screen the resulting models for safety using whatever tools we have, and only if we are convinced that the models underlying our alignment helpers are safe, specialize them to help us solve alignment.

This can be done by e.g. finetuning our powerful base-models on alignment research once we know via other techniques that they are safe, and monitoring this process very carefully. This overall seems much better and more cautious than just pretraining on everything including alignment research and indeed it seems likely that to create such a specialized research AI some kind of secondary finetuning step would be necessary anyway.

Beyond this, there should likely be a second designation of information that is too sensitive even to be shown to our aligned helper AIs. This might include exact evaluation and red-teaming protocols that we use to test our AGI systems (including such alignment-helper AGIs) as well as any other information that seems critical to the success of our methods of detecting misalignment.

2.) *Removing e.g. all ML research on arxiv might substantially impair the actual capabilities of the model in general -- i.e. it might impose too great an alignment tax. For instance, there is some evidence that general reasoning ability is helped by training on code data, and there might be a similar effect where ML papers are very high quality data and improves capabilities significantly*. I would be surprised if this is the case, at least for just LW/AF data and not ML arxiv since the absolute amount of text is small relative to the size of the training corpora. In this case, it seems reasonable to just take the hit of the alignment tax. If it truly is unbearable, then maybe something can be bootstrapped where we can use existing language models to generate plausible but subtly wrong alignment related data to train upon to further ensure and mislead potential misaligned AGIs.

3.) *We are already running out of data and high quality data sources such as arxiv and LW/AF are too valuable not to use for general pretraining.* I am generally least sympathetic to this point and it also seems incorrect in that with trillion-token datasets, LW/AF and other alignment relevant resources can only possibly comprise a tiny fraction of the total high quality data. People also already filter away other sources of possibly high quality materials for much less principled reasons such as removing NSFW or copyrighted content which have little bearing on safety but large bearing on legal liability.

4.) *Such an approach won't catch everything anyway and a superintelligent AI might be able to infer enough of the missing remainder such that it doesn't matter*. This may be true but I would argue we do not know for sure at this point, and to me it seems highly unlikely that it would have no or little impact. The cost is relatively low and the potential gain for safety is high so I am still in favour even if it may not work perfectly. Such an approach is made even easier by the high centralization of the relevant data -- removing just LW/AF and ML arxiv tags from the corpus would probably remove ~80% of the offending material.
