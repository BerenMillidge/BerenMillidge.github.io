---
layout: post
title: My Preliminary Thoughts on AI Safety Regulation
---

**Epistemic status** *Still trying to work out my thoughts on this. Things change pretty regularly. My current thinking on technical AI safety questions and threat models likely diverges by now reasonably far from the LW median.*

I have been thinking a little recently about how I would approach AI regulation, if I was in the position of a regulator or some kind of lobbyist (which I am not and generally do not plan to be) because it is an interesting way to ground my current views on what the main potential harms are and the concrete things we can do now to nudge the world towards better futures, on net. I don't hold any of these views super strongly and they evolve fairly regularly both as new evidence comes to light and as I think about things more deeply. My recent desire to write and post something was driven by the recent move by India to require approval to deploy any AI generative model. I think this is probably over broad and will probably hurt safety in the long run if similar regulations are brought into many other countries since similar regulations will just entrench existing AI interests who have a mixed safety record and, in any case, cannot hope to solve the problem all by themselves.

In general, I think we are still quite early in the development of AI and while it is useful to be cautious we also don't want to completely stifle the technology nor cut off new entrants to the field by onerous regulation (in fact I think this would be bad and decrease safety overall). I think in general, the current focus should be on preventing the emergence of strong and autonomous agents that can self replicate, the development of robust auditing frameworks for frontier models, and dealing with misuse harms as they crop up without making any strongly decisive moves. I broadly do not think that existing generative models pose any significant existential threat since they currently appear to lack any kind of coherent agency or tendency to behave consistently adversarially to humans. Instead, I think these are quintessentially 'toolAIs' currently and that their development should be supported while work towawrds more explicitly agentic and autonomous systems should be carefully scrutinized. In my view, almost all danger comes from more explicitly agentic AI systems with long term memory and the ability to develop an independent personhood concepts and survive and replicate independent of humanity. Such danger can arise in one of two ways -- either a centralized monopoly produces a misaligned agent which, because there is an AI monopoly, can overpower everyone else and hence cause doom or, alternatively, in a highly multipolar world, there comes to exist a darwinian population of replicating agents of varying degrees of alignment for which evolutionary pressures will push them towards implicit competition with humanity. 

As such, the general principles I would put forward regarding any regulation at this point would be as follows:

1.) It is important not to make any decisive moves that would massively slow down development or entrench current incumbents in their positions of power. The field and market is currently extremely dynamic and there are many very positive developments in terms of funding flowing into AI right now as well as many breakthroughs coming from many sources. I feel that current AI technology has immense potential to be broadly beneficial to humanity and much would be lost if it was crippled by regulation in the same way as e.g. nuclear power or biology in general. I also think that a centralized development of AGI around a few government sanctioned companies would not be particularly safe, especially compared to a transparent and open development. I think that concentrating power in a few 'trusted' AI companies, as well as restricting information flow around AI development and AI safety research is [extremely likely to backfire](https://www.beren.io/2023-08-09-Strong-Infohazard-Norms-Lead-To-Predictable-Failure-Modes/), resulting in poor epistemics, stunted safety progress, extreme conflicts of interest, and a generally less safe world.

2.) I would not effectively disallow open-source models and development by heaping onerous requirements on all market participants regardless of scale -- as recently [happened today in India](https://twitter.com/AravSrinivas/status/1764429686787252402). I think requiring prior approval for all actions is massively onerous and is on a standard that few other industries follow and especially no healthy, non-sclerotic industries. I think that a [vibrant open-source ecosystem has caused the majority of the AI safety progress](https://www.beren.io/2023-11-05-Open-source-AI-has-been-vital-for-alignment/), especially in the last year, and this strongly looks to continue. On net, I would say that regulation, if possible, should broadly tilt towards favouring new entrants against incumbents since more decentralized power in AI seems broadly safer to me and is likely to lead to faster alignment progress.

3.) I would encourage the development of auditing methods and independent auditing infrastructures for frontier labs. I don't think current models show strong evidence for existential risk, but this may occur soon (you never know) and having such capabilities just seems generally useful for informing both the general public, including academics about the current state of the art, as well as informing regulators as to the actual capabilities and dangers of such systems. I would suggest that evaluations be created that test for such dangerous capabilities and would not be particularly onerous to comply with if possible. I would also establish a high floor such that only frontier systems would be subject to them and that this floor is moved upwards over time along with frontier systems. I.e. if GPT4 is safe it is silly to have a licensing regime that would cover GPT3 tier models. I think having transparent and open evaluations of dangerous capabilities, however, is very important and is currently a large gap that should be filled. To me this seems like a clear public good and potentially could be filled by a non-profit or governmental organization if nobody else fills it. In general, my view is that anything that improves transparency is likely to be net good.

4.) I would push for more transparency from frontier labs about their alignment technologies. While there is a clear business case for secrecy about model internals, much alignment work appears to be less crucial and there are large benefits to transparency here for safety -- both that other labs can copy any advancements in safety and alignment that have been made and also that independent researchers and academia can interact much more easily with the state of the art. This transparency will also help reveal any shortcomings or defects of major labs strategies or alignment implementations which would otherwise be hidden by secrecy by default. While painful initially, this would result in a much stronger and more robust set of alignment techniques in the long term. Alignment research, in genreal, I think should be considered a public good and regulated as such.

More broadly, AI alignment seems to be a problem which is harder to achieve in a centralized manner than capabilities. Capabilities mostly seem to rely on concentration of compute and capital while requiring minimal novel intellectual breakthroughs while alignment is the opposite. Large and secretive corporations are extremely good at concentrating capital on well-defined engineering ends and are bad at making large numbers of intellectual breakthroughs. Thus, I think that concentrating all power and knowledge of frontier in AI systems in a few large corporations is largely bad for alignment relative to capabilities. In the long run, such a centralized system may also slow down capabilities when pure engineering and capital concentration reach the top of their respective sigmoids and fundamental research progress is needed for continued advancement, however it is hard to know where the limits of the current scaling paradigm are and they may be significantly into the level of existentially unsafe systems. 

5.) I would increase academic funding towards studying AI alignment research of varying types without a strong bias towards one approach or another -- i.e. interpretability over finetuning or RLHF or theoretical work. In general, I think that having lots of brains attacking the problem from different angles is good and I do not believe that grant making agencies are that good at assessing it. More broadly, getting alignment work to have a more sound standing within academia is very important to ensure its long term.

6.) More broadly, I see the current situation as one where it would be wise for states and other regulating entities to start gathering information, understanding and modelling the actual problem and potential situations and start building up the *capacity* for acting rapidly and decisively if the situation demands it. This would also be a good time to start constructing official communications channels between leading actors and regulators and institutions connecting them so that it becomes much easier to coordinate responses to threats in the future and to react rapidly if novel issues emerge. 

Notes on specific policies I have seen proposed:

I think compute caps in terms of flops (and similar KYC customers for hardware supply chain) seem overly crude measures which don't sufficiently discriminate between safe and beneficial uses of AI and the dangerous ones. While more flops certainly potentially enables more danger, it also enables more positive outcomes as well. In my view, the key dangers of an AI is autonomous agency and especially self-replication. I think that no current systems appear to approach that and for many systems this will never realistically be an option, regardless of the compute. For instance, I feel that video generation models such as Sora or protein folding systems like AlphaFold, although they undoubtedly use huge amounts of compute, are fundamentally safe regardless of the compute they use. Alternatively, cleverly trained agents specifically targeted towards such capabilities may be feasible at compute budgets significantly below e.g. GPT4.

While a KYC infrastructure for compute may be useful for general information gathering, I would be wary based on the bureaucratic incentives for such an agency to start writing and enforcing regulations before this should occur and could be useful. In general, a premature slouching towards regulation by pure bureaucratic entropy as occurs in many other fields seems to be unhelpful at the moment.

For misuse risks I would be cautious and see whether existing cases of clear misuse can instead be regulated and prosecuted under other existing legislation rather than AI specific ones. This is almost certainly the case since most current harms from AI misuse fall into already clearly existing criminal brackets. I.e. using AI voice cloning for scam calls falls clearly into laws regarding scams in general, using AIs for bioweapon construction falls into already existing bioweapon regulations. More generally, I think that the current regime is one of toolAI and should be regulated as such where AI technologies simply enhance and extend human capabilities rather than present novel or existential threats. I think generally setting up government departments and institutes for studying such risks is probably a good move, since states will need the capability to effectively keep up with state of the art and assess new and existing risks in the near future.

I think a full AI pause as some people advocate is not strictly necessary at this point as to me there are no immediate existential threats posed by current systems. This may change in a few years or longer. I think that, however, having the *ability* to force a pause is potentially useful and having that option in the Overton window is potentially a valuable thing for people to push for, to be used if it is needed. I think it is possible that there is some left-field breakthrough that makes systems significantly more dangerous in which case a pause would be advisable, however in the default world path I think we are on, I don't think a pause would be warranted in the near future.

I think [responsible scaling policies](https://www.anthropic.com/news/anthropics-responsible-scaling-policy) as advocated for by Anthropic are sensible. In general, I think having clear and transparent metrics about the dangers of systems as well as broadly agreed upon guidelines for when to pause scaling or other capabilities development are desirable and should be table-stakes for people serious about safety. In the ideal case, there would be a set of such guidelines and policies that all frontier labs would agree upon and that convening the required meetings and having the leverage to force such an agreement and that the actors involve stick to it at moments of crisis would be a very sensible use of government power. The key to the effectiveness of such policies is whether the actors involve will actually abide by them when models start becoming potentially dangerous and there is strong competition, including internationally. Having a set of powerful and credible precommitments including both internal organizational incentives and external organizations with the power to force commitments like these would strengthen the current setup.

I am generally in favour of mandatory evaluations as long as they are not onerous on new entrants and actually measure useful safety-relevant properties -- i.e. something like today's LM-eval and model cards being required I am okay with. Something like filling out a billion forms and waiting a year to hear back from some overloaded government agency with a presumption of rejection is much less good. I am generally opposed to restrictive licensing regimes based on use-case or compute or general deployment since they tend to entrench incumbents.

In general, while we should be cautious and safe, at the same time we must not forget that AI has the potential to bring about immense good for humanity and ultimately the next stage in our evolution up the Kardashev scale. Fundamentally, the goal has to be to use AI to max out technology, cure ourselves of our current ills and limitations, and transcend our biological substrate. We should be careful about risks but our risk tolerance should not be zero. Sitting forever on this planet with a totalitarian government dedicated to squashing AI (and ultimately all technology since any kind of transhumanism has its own intrinsic 'alignment problem') forever is not a stable equilibrium. We can, and perhaps should be slower and more cautious but at the same time the opportunity costs are nontrivial -- many millions of people die of aging every year, and day by day the universe expands and galaxies drift beyond the cosmological horizon. While it is relatively easy to set a bureaucratic ratchet in progress, as has been demonstrated ample times in recent history, stopping or reversing it is extremely hard. EA and AI safety lobbyists are swimming with the current here so we should be careful to celebrate successes but also to fight against unhelpful and counterproductive overreach. The main thing I hope is that decisions are made and policies are crafted with a sensible and realistic understanding of the actual sources of existential risk that AI poses and a specific model of what must be addressed rather than just reacting to the events and hysterias of the day. 