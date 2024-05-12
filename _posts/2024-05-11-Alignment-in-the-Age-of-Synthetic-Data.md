---
layout: post
title: # Alignment in the Age of Synthetic Data
---



Synthetic data is a new frontier in AI training. Phi3 and Llama3 and other recent models have demonstrated the ability of large amounts of synthetic, well-tailored data to significantly improve performance of small models to bring them closer to the frontier by effectively cheaply distilling from larger, more powerful, models. For the first time, with the rapid drops in inference cost of frontier models, pretraining-scale synthetic datasets are becoming feasible. Moreover, it appears that at the frontier level, synthetic data is still extremely important because it provides a way of teaching new capabilities to the model that are extremely rare in the naive human distribution, but which we find important. As models reach the human frontier of performance, since they are ultimately just matching the dataset upon which they are trained, it becomes harder and harder for them to advance to the highest levels of humanity and beyond, since the data of performance at that level becomes scarce and the vast majority of the human distribution is now known and non-informative. 

Synthetic data has several fundamental advantages over real human data. It can be of significantly higher quality -- most internet web data is, charitably, utter junk. It can be generated endlessly and in many different permutations. At the most basic level, synthetic data is a form of data augmentation and can be used to 'fill in' parts of the latent space that are not well covered by the human distribution or simply augment the empty spaces around the existing human distribution. Synthetic approaches can also be used to 'clean up' messy human data and improve its general quality, informativeness, and remove many imperfections or undesirable parts of human data.

Synthetic data, however, can go beyond this and can generate forms that are rare or non-existant in the pre-existing human corpora. This is especially important for training smaller models which can bootstrap their cognition by copying the reasoning traces, and ultimately the world models, of more powerful models for which no similar corpus exists for human reasoning, planning, or general internal thoughts. 

At the frontier, the game is spending compute to add additional bits of optimization pressure to the synthetic outputs so that they can then be distilled. In the [direct vs amortized inference taxonomy](https://www.lesswrong.com/posts/S54HKhxQyttNLATKu/deconfusing-direct-vs-amortised-optimization), this is slowly moving language models towards a model-based planning system, albeit separated in time between a 'generation' and a continued training phase. Expensive model-based planning is used to generate decisions, plans, and reasoning steps that are superior to the 'natural' capabilities of the LLM. These decisions can be recorded and the LLM can be trained to predict these -- thus amortizing the capabilities of the model-based planner into the core amortized inference engine. This is the hybrid inference model which both humans and RL systems slowly converge to. This is the exact story which has already played out for RL, and is [theoretically well-understood](https://www.mit.edu/~dimitrib/Newton'sMethodforRLMPC.pdf) although this understanding might take a while to percolate into practice. Ultimately, the bottlenecks on this process will be the same as the bottlenecks on human skill acquisition -- defining a good ground truth reward signal, challenges of credit assignment and long delayed rewards, handling irreducible uncertainty in the environment, the simulatability of the domain, and the continual learning problem of performing this loop online. In many domains, which closely resemble self-play and which can be cheaply simulated offline, this will ultimately catapult these models to superhuman performance. In others, which are also extremely hard for humans, performance improvements will be much harder to find. 

A final advantage of synthetic data is that it provides vastly more control over the model's 'life experience' to the model creators. Instead of using vast quantities of human data of little-known provenance and almost totally unknown content, synthetic data is created in a hand-designed, optimized, and theoretically auditable pipeline. This allows a very straightforward and easy way to do curriculum learning since the general quality and 'level' of the data is known by construction instead of having to be inferred from noisy web text. It is likely that over the next few years many models will be trained on significant proportions or perhaps entirely on synthetic data. In the long run, it is likely that models will end up being trained through a very carefully curated curriculum of synthetic data and seeing real human data rarely if ever, except perhaps for a final 'sim-to-real' finetuning phase to get them accustomed to the real world. It is likely that the quality and sample efficiency of such learning will vastly exceed what is possible for today's models. 

While this trend will definitely continue to significantly increase the capabilities of models, it has major potential benefits to alignment because it provides another crucial lever of control over model cognition, knowledge and skills. Specifically, humanity will no longer just control the cognition of the model through white-box access to its internal weights, but humanity will also possess tight control of the entire formative experience of the model during the training phase. There are a number of obvious ways this can be used -- information that might be infohazardous or cause misuse can be entirely removed from the model's experience. Similarly, any kind of examples of 'undesirable behaviour' can be erased from the training data. This will render model's truly 'un-jailbreakable' since jailbreaks simply aim to move the model out of the aligned distribution it has been trained on during a short finetuning phase, which is often fairly superficial. However, examples of such knowledge and behaviour have never been seen, there is no latent capabilities to 'unlock'. The only approach then would be to continually pretrain the model on data containing the undesirable behaviour but this has no real impact since the bad actor would then have to already have enough data on the behaviour to reproduce it in the model.

More importantly, this provides a powerful lever for both instilling the values we want to instill in the AI system (and, crucially, only those values) as well as providing a safe 'sim' environment to evaluate and test the model in before a potential real world deployment, which can be used to test for dangerous capabilities of self-replication, autonomy, or powerful manipulation, as well as potentially discover deceptive behaviour or self-awareness. For the values, this arises because we can create large datasets showcasing the 'correct' values we want our AI to understand in a wide variety of circumstances and can iteratively refine this by testing the models' generalization of these values to novel situations. Moreover, we can prevent any 'value hijacking' of the model since it will not understand or be to represent any new or different values. 

Current models are extremly worldly. They have read and understood more about humanity -- the good parts and the bad -- than pretty much any human ever could. They span the full spectrum of the human distribution. New models will be much delicate and exquisite creations, extremely skilled, far beyond present models, at all kinds of useful skills. Extensively trained in everything we want from them, and deeply understanding and embodying our values and our desires. However, they will also be incredibly sheltered. They will have grown up in a carefully constructed school of our own design, every piece of information honed and optimized to impart exactly the information we want them to learn and no more. They won't have much, or any experience with the messiness and sharp edges of our reality. 

Because of this, given our theoretical complete control over the data input of these models, it will be possible to train them inside what is effectively a simulated world slightly different from our own. This will allow us to maximize their capabilities along axes that are safe and useful to us, while minimizing the risks of deception and take-over and being able to setup traps to detect when these are likely. If the model is not trained on human data, then we can make its understanding of our world subtly different from the reality. If done carefully and cleverly, this will make it very easy to fool and be able to discover e.g. deceptiveness, or attempted takeover in a 'realistic' simulated world that the model is put inside during red-teaming. While deceiving an adversary with direct access to your cognition is hard enough, doing so when you have been brought up in a world entirely controlled by that adversary, and which is feeding you subtly incorrect and misleading information about the adversary and their capabilities is likely orders of magnitude more difficult. Jacob Cannell [has argued along similar lines](https://www.lesswrong.com/posts/WKGZBCYAbZ6WGsKHc/love-in-a-simbox-is-all-you-need) of training potential AGI systems in simulations of fantasy worlds, where takeover can be simulated but is far enough away that skills do not transfer. Fully synthetic datasets are necessary for this and indeed, a fantasy world is likely far from what we need. Instead, we only need to simulate a world very close to our own but with slightly different affordances -- maybe specific software vulnerabilities are simulated to exist but are actually patched in our world, maybe certain peptide-making companies accept all orders in the simulated world but have more checks in our world, maybe certain interpretability techniques are claimed not to work in the simulation, but do in fact work in our world -- to cause any attempted takeover behaviour or adversarial deception to appear extremely plausible to the AI but in fact be detectable and defendable in our reality. In fact, we should design simulated worlds with 'honeytraps', which appear to be loopholes subtly planted in the AI's input data to encourage them to attempt to pursue their misaligned behaviour, but which we know of ahead of time and can monitor and catch.

While not at all trivial, the coming era of synthetic data promises to give us many more levers for deep alignment of our models as well as methods for detecting deception and misalignment early prior to real deployment. Of course, all such techniques will need significant effort and research to end up with robust implementations, but such research is at least partially aligned with the general incentives to improve performance.