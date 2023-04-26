---
title: LLMs confabulate not hallucinate
layout: post
---
*Minor terminological nitpick.*

People often describe the LLM as 'hallucinating' information whenever it makes up information which seems like it should fit for a given query even when it is trivially false. While evocative, this isn't actually correct terminology. We already have a perfect word for this precise phenomenon in psychology: confabulation. 

Confabulation is typically used in a psychiatric context when people have some kind of brain damage, especially to memory, which causes them not to be able to explain or answer questions correctly. For instance, if an amnesiac patient is asked questions about an event they were previously at, instead of admitting they do not know, they would invent a plausible story. Similarly, in split-brain patients, where the corpus callosum is severed so each half of the brain cannot talk to each other, patients can invent elaborate explanations for why the other half of their body is doing a specific thing, even when the experimenter knows this is not the case because they have prompted it with something differently.  In general, people confabulating invent plausible sounding justifications which have no basis in fact. This is usually not conscious with an intent to deceive but instead appear to strongly believe in the story they have just reported. This behaviour is identical to what LLMs do. When they are forced to give an answer using a fact they do not know, they cannot say that they don't know, since in training such examples would be followed by the actual fact. Instead, they make up something plausible. They confabulate. 

More importantly, if we recognize that what LLMs are really doing is confabulating, we can try to compare and contrast their behaviour with that of humans. Humans confabulate in a wide variety of circumstances and especially with some neural pathologies such as with memory impairments and in split-brain patient. Funnily enough, these are similar to a LLM. What are LLMs but humans with extreme amnesia and no central coherence?
