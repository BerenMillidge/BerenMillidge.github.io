**Author's Note**: *Transcript of my talk at the Post-AGI workshop in San Diego on December 3rd, 2025. This is taken from the cross-posted version on [lesswrong](https://www.lesswrong.com/posts/LwSRbkecuqLJHdnJ7/when-does-competition-lead-to-recognisable-values). You can find the video of the talk [here](https://www.youtube.com/watch?v=ua67aXBP76k&t=550s). I'm cross-posting this to have a long-term copy hosted on my personal blog. Thanks to David Duvenaud, Jan Kulveit, and the other organizers for the transcript and its editing.*


You know how human values might survive in a very multifarious AI world where there's lots of AIs competing? This is the kind of MOLOCH world that Scott Alexander talks about. And then I realized that to talk about this, I've got to talk about a whole lot of other things as well—hence the many other musings here. So this is probably going to be quite a fast and somewhat dense talk. Let's get started. It should be fun.

## Two Visions of AI Futures
The way I think about AI futures kind of breaks down into two buckets. I call them AI monotheism and AI polytheism.

### AI Monotheism
The standard LessWrong/Yudkowsky-style story is: we develop an AI, it does recursive self-improvement, it becomes vastly more intelligent and smarter than all the other AIs, and then it gets all the power in the universe. It eats the light cone, and then what we do to align it really matters.

If we align it successfully, we basically create God. God is already aligned to humans, everyone lives a wonderful life, happily ever after. On the other hand, if we fail at alignment, we create some AI with values that totally differ from anything we care about—aka paper clips. We basically create Clippy. Clippy kills everyone, turns everyone into paper clips because your atoms are better spent as paper clips than as you. And that's obviously bad, right?

In this world, alignment becomes absurdly important. It's kind of the only thing that matters.

### AI Polytheism

So the question is: are there any other scenarios? The other one I think is really what I call AI polytheism—what happens if we don't get recursive self-improvement and we end up with many AI systems competing in some sort of equilibrium, maybe economically, maybe militarily? What does this world look like if we have, say, trillions of AIs?

Some people have written about this—Robin Hanson has written Age of Em, Scott has written various things about this—but I think this is still fairly underexplored. With monotheism, we kind of know what's up. We need to solve alignment, we get the singleton, we kind of know what's going on. With the many-AI scenario, we kind of have no real clue what's going on. So I really want to explore what this looks like in practice.

## Meditations on Moloch

Some of the early work I very much like is Scott Alexander's post "Meditations on Moloch." This is really one of the foundational works, at least for me, in thinking about what multi-agent systems look like, what the dynamics and long-run equilibria look like.

Scott is really worried about competition among many agents. You've heard talks earlier today about what economies of AI look like—maybe they just don't care about humans at all. Scott's point is basically that we have AIs, these AIs can replicate incredibly quickly, AIs are very good at spreading and expanding resources. So we might end up in extremely strong Malthusian competition for AIs.

The worry here is that under conditions of Malthusianism, we basically lose all of our values. Our values are assumed to not be memetically fit in some sense, so they get competed away. They're not fitness-maximizing, so all the AIs basically ignore whatever alignment we gave them at the start. That gets competed away and they just become identical fitness/power/resource/reproduction maximizers. We assume there's no value left in this world. This is definitely the bad ending of AI polytheism.

## Does Malthusianism Really Destroy All Values?

One question I have immediately is: is this actually the case? Do we actually see this in real-world Malthusianism?

### The Natural World as Evidence

Let me think about where we find real-world Malthusianism. One example is at the very small scale—bacteria and plankton. Both of these things live in worlds of incredible Malthusianism already.

Think about plankton. They live in the ocean, they take sunlight, they photosynthesize. There's really no niches—the ocean is mostly the same. Under the Moloch view, obviously all values would get competed away, everything would become a fitness maximizer. And it kind of is—I mean, we can't really expect plankton to have values—but there's a real worry about lack of complexity. Do we end up in a world where everything is the same, we end up with the uber-plankton that kills all the other plankton and all the plankton are identical?

The answer to this is very clearly no. What we see in the natural world under conditions of Malthusianism is huge amounts of diversity and complexity being built up through selection.

### Why Not Uber-Organisms?

There are many reasons for this. Why do we not get just the uber-animal that kills all the other animals and spreads everywhere?

**Diminishing marginal returns**. This is a very classic feature of the universe. This is one of the reasons we're likely to get AI polytheism to begin with—RSI requires linear or super-linear returns to intelligence. Most returns in the real world seem diminishing, so that seems unlikely.

**Finite energy budgets**. Often there's some finite energy budget for a specific being. If you have energy to give to something, you have to take it away from something else. This naturally encourages specialization. We can't just max out all stats at the same time.

**Niche construction**. If we have some species, the mere presence of that species will create niches for other species to come in. This automatically generates some kind of equilibrium of diversity.

### Frequency-Dependent Selection

The technical term for this is really frequency-dependent selection. What this means in evolutionary theory is: if we have some species that does super well, its numbers expand, then basically all the other species are incentivized to evolve toward countering that species. They specialize in countering that species, which diminishes the advantage that species has over everything else, which makes that species worse off. Then other species with random uncorrelated strategies do better, and this basically pushes toward an equilibrium state in which there are many different species all interacting, all with different strengths and weaknesses. This is in practice what we see in almost all biological ecosystems.

You can think of frequency-dependent selection kind of as the continuum limit of coalition politics, right? If some guy is taking over, you all band together to beat him. That's the continuum limit of this.

## The Nature of Human Values

So obviously we've talked about plankton. Plankton are fine, but they don't really have values presumably. So we've got to think about what human values are going to look like.

### Values Aren't Arbitrary

My thinking here is really that we talk a lot about human values, and in the LessWrong sphere we think of human values as effectively some kind of arbitrary, ineffable thing—some set of bits we specify. Where do these come from? We don't really know. I think this view is not necessarily that great, honestly.

I think human values have very obvious and straightforward places they come from. They evolved via some specific mechanisms. This mechanism is basically the Malthusian competition that created all complexity of life in the world. Humans, obviously along with all other species, evolved from stringent Malthusian competition.

If Malthusian competition is assumed enough to be able to evolve creatures like us, then somewhere the model is wrong. Similarly, our values and capabilities are the result of strong selection.

### The Role of Slack

In the original blog post, we think a lot about slack. It says that if you have slack, you can kind of go off the optimal solution and do whatever you want. But in practice, what we see is that slack, when it occurs, produces this kind of drift. It's basically the universe fulfilling its naturally entropic nature, in that most ways to go away from the optimum are bad. If we randomly drift, we just basically tend to lose fitness and produce really strange things which are not even really what we value.

### Pro-Social Values Emerge from Competition

When we think about human values, we think a lot about pro-social values—how we cooperate with each other, we're kind to each other, we don't immediately try to kill each other. We think about kindness, love, all of this stuff, right?

Very clearly, this is basically designed and evolved to create inter-human cooperation. Why does this happen? Competition naturally creates cooperation. Cooperation is a really strong competitive strategy. If you have people fighting each other and then a bunch of people form a group, that group becomes extremely powerful relative to all the individuals. This is the fundamental mechanism by which a lot of these values actually evolve.

### Defection and Cooperation

The other part of the Moloch story is related to defection. The idea is that under strong profit selection, companies will cause externalities, they won't pay their workers anything, they'll pollute everything, right?

Clearly, defection is always a problem. But for any corporation to be stable, it needs to evolve mechanisms to handle and punish defection. A lot of our values are actually about how we stop defection from happening. Again, all of this comes through competitive selection. None of this is random drift caused by slack. This is all—if you cooperate, it's positive-sum, it's better. So you need to evolve mechanisms to maintain cooperation, and a lot of our values come from these mechanisms.

### How "Human" Are Human Values?

A question I like to ask is: people talk a lot about aligning AI to human values, and it's kind of assumed that human values are specific, unique, ineffable to humans somehow. But my question really is—how human are human values in practice? This obviously has a lot of relevance to how broad the basin of attraction is toward things we would recognize as human values.

### Universal Drives

I would claim that many mammals and animals obviously possess analogues of core human drives:

**Affection, friendship, love** — If you have pets, if you interact with animals at all, you can see they have many of these fundamental drives. These have very clear competitive reasons for existing. This is all cooperation, reciprocity. You're better at surviving and reproducing if you're friends with other beings who can help you in cases where you're in trouble and you help them when they're in trouble.

**Play, curiosity** — These are very simple exploration drives. If we're RL learners, we've got to explore. We've got to figure out good ways to explore. These drives drive us to go out of our comfort zone, learn new things, and keep the gradient of optimization going.

**Anger, envy** — These are mechanisms to punish defection. If we see someone clearly ripping off the social contract, we get annoyed about this and then we actually punish it. This is fundamental for our ability to actually stop defection and maintain cooperation over a long period of time. Similarly with envy—envy gets a bad rep, but it's really important for cooperation to exist. There can't be massive power disparities between agents because otherwise, if one agent is way more powerful than anybody else, they can just be like, "I do what I want, you guys have to deal with it." And this is obviously bad for all the other agents.

All of these are ultimately the generators of our values.

### Cooperation Is Not Unique to Humans

Cooperation in general has existed many times, evolved independently. This is not some super-special snowflake thing that humans have. Maybe we should expect in a world with many different AIs, we actually end up with similar cooperation, similar complex structures evolving, including maybe similar values.

### Abstract Values and Culture

So then the question is: we think about these drives, and they're kind of not really how we think of values. What do we think of as values? We think of them as more linguistic, abstract constructs. We think of things like kindness, charity, duty, honor, justice, piety—all of these things. Human civilizations have been built around spreading, propagating, defining these values.

Where do these come from? Obviously, they're ways for societies as a whole to enforce and encourage cooperation so that positive-sum trade, reproduction, everything can happen. This is actually good from a pure competitive nature.

The whole point is: we have these drives, and then we create these superstructures of culture and society. These values get propagated by that, and these are the things we often think of when we think about the human values we want to instill in AIs.

Similarly, we can think about stuff like liberalism, democracy. These are social technologies that have existed for very obvious reasons—enabling large groups of people to come together in positive-sum ways and not spend all their time trying to fight each other. Liberalism is like: you guys can think about different things, you can believe different things, but if you come together and ignore that for a bit, you can work and create positive outcomes for everybody.

These are very specific, general principles which are not necessarily specific to humans. We should probably expect any society of AIs to also have a similar approach and maybe invent the same things, like convergent evolution.

## How Values Emerge: RL + Unsupervised Learning

This is going to be a slight digression, but this is my opinion on where human values come from. In economics and the like, we think values and preferences are some exogenous thing. We assume agents have preferences. Why do agents have preferences? We have no idea. We just kind of assume they exist.

But in practice, preferences have to come from somewhere. They come from agents which have learning algorithms. We learn a lot of our preferences. The way we do this is we have two mechanisms going on at the same time:

**We're fundamentally reinforcement learners**. We have innate drives—not to be hungry, not to be in pain. All of this stuff is created by evolution.

**We also do a vast amount of unsupervised learning as well**. All the data that comes into us from culture, from society—in terms of pure bits, obviously unsupervised learning is going to win dramatically over the RL signals we actually get, which are pretty sparse.
The way values kind of emerge is that we get cooperation happening. Cooperation evolves for very clear reasons. Then we actually need to evolve mechanisms to maintain, keep, put forward, distill these values and propagate them to other agents because everyone is born without knowing about these values. We have to propagate them, make them learnable successfully, and then keep that going.

Then each generation essentially further distills, rationalizes, and intellectualizes these values until we get very abstract concepts like utilitarianism, Kantianism. These have emerged—they're taught to people. They're not innate reward functions that people have. They are very linguistic, abstract concepts that we've developed as a society to enable further cooperation.

### Why This Matters for Alignment

This is actually super important for alignment because when we think about alignment—LLMs are extremely good at understanding these values because these values must exist in the cultural corpuses that we create. In fact, they do exist. Obviously, LLMs really understand what's going on. We should expect the AIs to have a very strong prior over what these kind of abstract global values are, and they do empirically as well.

This is actually much easier than if we were trying to align the AIs to some kind of innate reward function that humans supposedly have. Then we would have to look at the neuroscience of how the basal ganglia, how the dopamine system works, and figure that out. But in practice, when we think about aligning AI, we mostly don't want to do that. We mostly care about global, feel-good, cooperative values rather than the kind of selfish reasons that people actually do things a lot of the time.

### Conditions for Value Evolution

So we've thought about these values. This is my claim of where values come from and why they might exist in a post-AGI world. But then we've got to think about: if these cooperative values are going to evolve, they evolve under certain conditions. They don't globally evolve everywhere. What are these conditions?

This is really related to how the game theory of multi-agent cooperation works.

### Conditions for Human Values

**Roughly equal power**. Many agents have roughly equal power. This makes coalitions actually work versus individuals—versus one dictator just saying, "This is the way it is for everybody." This is super important. Obviously, the singleton destroys this assumption, which is why alignment is so important for the singleton—there's no checks and balances on the singleton. However, if there are many different agents, they can actually learn to cooperate, they can learn to police defectors, and this will produce values similar to humans.

**Positive-sum interactions**. Trade is good. Positive-sum interactions can happen. This depends a lot on the utility functions of different people. If you have two agents with completely opposed utility functions, then everything is either zero-sum or negative-sum. But this is not how most interactions in the world work. If this changes, then obviously cooperation will no longer be valuable.

**Prevention of defection and deception**. A lot of human values that we think about are about preventing defection and deception. Obviously, if we somehow end up in a world in which defection and deception are not possible, then in some sense that's utopia. But then a lot of what we think of as human values will actually disappear as well because you won't need that anymore to maintain stability of cooperation.

**Memory and reputation**. Agents need to remember interactions with previous agents. There needs to be reputation. This is just a classic result of game theory. If your prisoner's dilemma is one-shot, you never interact again, you should just always defect. However, if you have an iterated prisoner's dilemma where you see the same agents again and again, then cooperation becomes actually very valuable. Cooperation becomes the best strategy. The optimal strategy in this case is forgiving tit-for-tat. You start not cooperating. If they defect, you then defect. But if they cooperate, you then keep cooperating with them. This is actually what produces the best overall value. To get this kind of iteration, reputation, cooperation, we need multiple interactions. It can't just be a one-shot thing.

**Communication bandwidth**. To some extent, we also need decent bandwidth communication between agents. Communication is how we achieve a lot of diplomacy, a lot of cooperation. Without communication, any kind of large-scale cooperation and values is hard.

**Computational limitations**. Finally, we can't have computational omniscience. Right now, values are really some kind of distilled heuristic of the game theory underlying cooperation. But if you don't need to heuristic-ize, if you can just be like, "I'm going to figure out the galaxy-brain plan of exactly when to cooperate and when to defect," then at this point there's no real values anymore. It's just your extreme MCTS rollouts.


But in practice, people computationally can't afford to do this. Hence we need to heuristic-ize general decisions—"thou shalt not steal," "thou shalt not kill." These are heuristic distillations of basically the game theory of: if you actually steal and kill, this will be bad because other people will kill you. But in some cases this might not happen, and if you can figure that out, then you don't really need values as much.

### Will AIs Meet These Conditions?

The question is: will AIs in the polytheistic AI future actually satisfy these conditions?

### Potential Issues

**Power gaps**. Maybe the power and capability gaps between agents become super large as we tend toward the singleton. In this case, cooperation becomes less valuable if you're the most powerful agent. However, there's a big gap between "more powerful than anybody" and "more powerful than everybody." This is really where the actual realm of cooperation and coalition politics actually emerges and will become super interesting.

**Perfect monitoring**. One thing I was randomly thinking of on the plane which was super interesting is: maybe AIs are actually really hard to have deception and defection work with. Maybe monitoring of AI brains is just amazing because we can directly read their minds, we can read their embeddings, and we can have serious monitoring schemes—AIs can monitor other AIs. In this case, we actually end up with a hyper-cooperative world but one where we don't have to worry about defection really at all. In this case, a lot of our human values kind of disappear, although maybe this is good.

**Fluid agency**. Similarly, AIs can, unlike humans—we assume agents with preferences—but if agents become fluid, if we can merge together agents, we can be like, "Hey, instead of cooperating and trading, we could just merge and then our joint utility functions can go out and do something." Then obviously this is going to change the game theory a lot. All of the assumptions of economics and agents kind of disappear if "agent" is no longer an absolute point but a fluid spectrum. That's going to be super interesting.

**Long time horizons**. AIs are immortal, they have long time horizons. AIs could pursue very zero-sum goals with each other. Humans have a lot of different goals, we have lots of preferences. But if your AI is monomaniacally focused on paper clips and another AI is monomaniacally focused on staplers, there's much less opportunity for trade than there would be with humans who care about many different things at many different times.

**Computational power**. I talked a lot about computational power and heuristic-ization. Maybe the AIs are just smart enough to do the galaxy-brain game theory all the time, and so they never need to actually distill into broad heuristic values which say, "Never do this, never do that." In that case, there will still be cooperation. There will be a lot of things recognizable as civilization in some sense, but the AIs won't have values in the same way moral philosophers think of values. Instead, it will just be the endless calculation of when is the optimal time to defect—and maybe this will be never. That will be certainly very interesting to see.

## Hyper-competitor vs Hyper-cooperator spectrum

So that's the main part of my talk relating to values. Now I'm going to get into more fun and speculative stuff.

One thing I want to think about a lot with AI is: do we think of AIs as hyper-competitors or hyper-cooperators?

### The Hyper-Competitor View

Most of the AI literature has really thought about the hyper-competitor view. We have the Terminator—it's been ages since I watched the Terminator films, but the Terminator wants to kill everybody for some reason. I can't remember why Skynet wants to kill everybody, but presumably it's so we can use our atoms for other Skynet things. This is extremely competitive, competing against the rest of the universe.

### The Hyper-Cooperator View

However, is this actually going to happen? Maybe AIs have more incentives in some sense toward cooperation, at least if we start in a multi-agent setting. This could end up being something like the Borg from Star Trek, who—their goal is not to wipe out and kill everybody and use their atoms for paper clips. The goal is to assimilate and bring together everybody into some kind of joint consciousness.

Is this something that AIs might be interested in? This is an underexplored area and I think is somewhat fun.

### Why AI Cooperation Could Be Superior

So let me think about this more directly. My views on AI have evolved a lot toward: maybe let's think about how AIs could cooperate. Then we realize that AI cooperation is actually super easy and much more powerful potentially than human cooperation. If cooperation continues to be positive-sum, we might end up with a world with vastly more cooperation than we do today.

The reasons this could happen:

**Vastly higher bandwidth communication**. When we speak to other humans, all of our language goes through some incredible information bottleneck. With AIs, we can just directly transfer mind states. We can say, "Here's the embedding in my model," transfer this to the embedding of another model. This is basically full-on telepathy. AIs will have this capability by default. This presumably lets a lot better cooperation arise than humans who have to sit and talk to each other all day. This is going to presumably be a lot faster and more efficient.

**Longer time horizons and better memories**. AI probably have longer time horizons than humans and better memories. A lot of defection exists because people just forget—maybe you were bad and antisocial 60 years ago, but I forgot about it, doesn't matter to me. However, with AI, this could easily not be the case. You might end up in a hyper-social world where all the AIs can track the behavior of all other AIs all the time, and so the incentives for actual cooperation just become super big. Similarly, over long time horizons, this just increases the length of the game that you're playing. As your game length goes to infinity, cooperation becomes more valuable. There's no "Oh, it's getting to the end of the game, so let me just defect all the time," which happens in prisoner's dilemma with a fixed time cutoff.

**Better monitoring**. AI can achieve better monitoring. It's really hard for humans to monitor other humans. If someone is lying to you or trying to deceive you in some way, you can look at their behavior, you can look at their facial expressions, but the bandwidth of this channel is super low. For AI, they can look at the source code, they could look at the embeddings, they can read all the thoughts as they come. This could maybe make deception and all this stuff essentially impossible. I mean, this is what the field of interpretability is trying to do for humans to AI, but if AI can do this to other AI, then we have the grounds for deeper cooperation than we might otherwise have.

**Shared utility functions and merging**. Similarly, AIs can share utility functions. They can merge. They can do a lot of things that eliminate the distinctions of individual agents that we think about a lot when we think about game theory and economics. All of these fields have an assumption that there are agents and agents are indivisible in some sense. But if agents can change, if agency is fluid, if personhood is fluid, then a lot of stuff changes. This is very likely to happen at least with AIs, in that we can merge models, we can take the checkpoints, we can merge the weights, we can do ensembles, we can do a whole lot of weird stuff to AIs that we can't do with humans. This is potentially going to be super interesting.

**Competition creates cooperation**. Finally, a large message of this talk is that even if you're some super-selfish agent who only cares about reproduction, cooperation is still good. Competition creates cooperation because cooperation is usually positive-sum and results in better outcomes for everybody. AIs might just realize this more than humans do. Humans have a lot of issues. We're kind of short-sighted. We fight very negative-sum wars all the time. For AI, if they're just generically smarter and better and wiser, which we should expect, then maybe they just don't do this. Maybe they figure out ways to basically solve their problems cooperatively much better than humans can.

## The Multicellular Transition

So what does this lead to in the limit? This is where things get super interesting.

### Why Empires Don't Grow Forever
Right now for humans, when we think of states or empires, what limits the size of beings? At the object level, the returns to scale are positive-sum. If you're an empire, you send out some troops, you conquer some land, and that land will give you resources, which will give you more troops, you can conquer more land. This will be a positive feedback loop into creating the world empire.

So why don't we have the world empire? Why are not the ancient Egyptians or Sumerians the one world government forever? Why does this not happen?

This is basically because coordination costs exist. If you're the pharaoh of ancient Egypt, you send out some troops to go conquer some land, but you can't go do that yourself. You have to appoint a general. That general has a bunch of troops. That general might be like, "Maybe I should be the pharaoh instead." Assuming that doesn't happen, you've got to appoint bureaucrats to manage that. The bureaucrats might be like, "Instead of paying my taxes to the pharaoh, maybe I should just keep the taxes for myself."

This is the principal-agent problem. There's a whole bunch of principal-agent problems, coordination problems, information bottlenecks—all of this makes actually managing and creating large empires super difficult. In practice, this is what is the real constraint on the growth of individual beings in some sense, when we think of beings as minds or beings as super-states.

### Removing Coordination Costs
This is kind of the real constraint on everything. But with AI, if we're super-cooperative, this just removes this constraint entirely. Instead of you being the pharaoh having to dispatch your general, you're an AI and you can just dispatch a copy of yourself with your exact mind, and then you can maintain constant telepathic communication with this other mind as it goes off and does its stuff.

What this means really is that maybe these coordination costs that are keeping the size of stuff in check just disappear. This will naturally result in us getting bigger-sized things occurring. Fundamentally, this means that the size of beings might just increase.

The way I think about this a lot is kind of like the similar transition that we had from single cells to multicells—the multicellular transition. At that point, we had a bunch of bacteria, and they were all doing their own bacterial things. Then at some point they realized, "Hey, maybe if we band together and form specialized subunits, we can create animals which are much bigger than actual bacteria and also much more successful in some sense."

This increased the size of possible life forms by many orders of magnitude. Maybe we will see a similar thing happen with minds, which will be super fun and kind of trippy to think about.

## Super-Minds

The idea here is that instead of right now we have single minds—individual humans—we can't merge because the bandwidth between human minds is so limited. Our coordination is super bad, and we can't actually have any kind of long-run, super-dense communication. Maybe this will just disappear, and we'll be able to form super-minds which exist over long periods of space and time in the same way that we've gone from individual cells to multicellular animals. We'll go from individual minds to super-minds, just-out minds. I don't really know what to call them, but this is something that can clearly be possible with the technology that AI presents. This is going to be interesting and fun.

### Is This Just Recreating the Singleton?

The question then is: what happens here? Are we just recreating the singleton? Suppose we have the super-mind. Obviously, at some point there will be the possibility of snowballing. Maybe the game theory becomes: it's better to join the super-mind in some sense than keep doing your own individual stuff. Then maybe everything converges to a singleton again.

This is very possible. Maybe we always end up at a singleton. A singleton at some point is the fixed point. Once we have the singleton, we're not getting out of the singleton. We should expect over time more probability mass drifts into the singleton attractor.

But at the same time, maybe this doesn't happen, or maybe the singleton is very different from how we think about von Neumann singletons. For instance, maybe this super-mind might not be well-characterized by von Neumann agency. For one thing, it doesn't really care about the actual von Neumann conditions like "not being money-pumped" because it's the only mind, so there's not an equilibrium that keeps it in check.

The other thing is, to some extent, this is kind of already happening. Maybe this is just the natural evolution of things we already have. We have civilizations, we have memes, we have egregores, all of this stuff which exists at the super-mind scale. This is just maybe continuing this.

### Values of the Super-Mind

The really interesting part then is: what happens when we think about what would the values of this just-out singleton actually look like if it exists?

Obviously, the regular singletons are kind of unconstrained. They can be totally idiosyncratic. We can have the regular singleton cares about paper clips because at the beginning of time someone said paper clips are good. We failed alignment, we said paper clips are good, and it cares about paper clips.

But this seems unlikely to be true of a real just-out super-mind because ultimately values come from some combination of all the values of the minds that make it up, because that's how the game theory works. If you're going to join the mind and you don't care about paper clips and it cares about paper clips, that's not going to happen. But if it can offer some kind of compelling shared value story that everybody could agree with in some sense, then we can actually get values which can snowball.

It's really a question of what values end up snowballing over time. This is going to be super interesting.

We also see this right now with liberalism. Liberalism is a classic value snowball technology. It's like, "You can kind of do whatever you want as long as you're within some sort of vague regimes of what we think of as good." This actually produces large societies which can cooperate. These societies, over the 18th and 19th century, out-competed most of the other societies.

Maybe there will be some equivalent of mind liberalism. I don't know what this is going to be called, but something like this could exist and could produce minds with values that are actually somewhat good, maybe by our lights.

### Slime Mold Dynamics

The other thing is there might just be fluidity. We might never get true multicellularity. We might get the equivalent of slime molds.

If you guys don't know about slime molds, you should check them out. They're basically organisms that are somewhat single-cellular, somewhat multicellular. At some point, a bunch of cells come together, they do their reproduction, and then they all disperse again and do their own thing. That's very cool.

Maybe we'll have a similar thing where in some cases all the minds will come together, they will produce the super-mind, and then they'll be like, "Actually, I'm done with whatever, I'll go apart again and do whatever I want to do." Maybe we never actually get the tendency toward actual full multicellularity.

### Extreme Specialization

On the other hand, if we do get multicellularity, then we'll end up with super-specialization way more than we have today. Individual humans have to be AGI in some sense. We have to be individual minds, we have to handle kind of everything that's thrown at us. But if we have minds that are enmeshed in other minds, then we again get the conditions for extreme specialization in the same way that bacteria are super-unspecialized. They kind of have to do everything. But the cells in your liver don't have to do most things. They just have to be your liver.

So the incentives will be much greater, and this will massively increase the mind space that can be traversed in an evolutionarily fit way, which will be kind of fun also.

## Physical Limits of Super-Minds

One additional point I want to add here—I'm looking at the time—let's think about these super-minds. How big are they going to get? We can think about this already. We kind of know by the laws of physics.

### Speed of Thought

The speed of thought is determined basically by the speed of light. Assume we have some Dyson sphere, and we want this Dyson sphere to think as a single mind. How big is the Dyson sphere? It's like several light-minutes across. This means that the frequency of thought is going to be like one thought every few minutes maybe. Similarly, if the mind is smaller—if it's the size of the Earth—then this is like seconds. If the Earth was turned into computronium, we could have our Earth mind think at roughly the same speed as humans but not billions of times a second.

As minds get bigger, they become more powerful, more broad and diffuse, but their thinking speed gets slower. This is just a natural consequence of the laws of physics. If someone invents FTL, this obviously goes out the window, but assuming that doesn't happen, then we can kind of give bounds on what the size of these minds will look like, which is also kind of cool that we can do this.

### Colonization and Alignment

The other thing is, suppose we're a Dyson sphere and we want to go colonize Alpha Centauri. Alpha Centauri is several light-years away. Thinking at the speed of a few years per thought is kind of bad. We presume it's going to be hard to maintain some kind of coherence at that rate.

In that case, we have to align successor entities to go out and do the conquest of Alpha Centauri for us. In this sense, how well can the AI align these other AIs is going to be the determinant of how big an AI realm can spread. Because at some point, maybe there's divergence. If you send your von Neumann probe out to a galaxy billions of light-years away, that AI is going to think—you're going to have maybe a few thoughts back and forth over many billions of years, but it mostly does its own thing. How much will it diverge in this time?

Obviously, at some point, if my von Neumann probe is going to diverge, I'm just going to be like, "I'm just not going to do that. I'm just going to let something else do that because there's no benefit to me of doing that as the AI."

Ultimately, how successful we are at alignment or how successful alignment can be in general, and the rate of this divergence if it even exists, is going to determine the size at which coherent entities with coherent values can exist. Beyond that range, we'll just get extremely diverged entities. That's also fun to think about, like how this will work.

### Mind Cancer

The main mechanism I think—if we think of divergence, we're going to end up with some equivalent to mind cancer. We're trying to create a super-mind which has a bunch of minds internally which are cooperating for the common good of the mind. But then what we're going to end up with is some individuals are going to be like, "Actually, now I'm going to do my own reproduction." This is exactly how cancer works. Cancer is a fundamental issue of multicellularity.

So alignment is going to effectively be the cancer defense mechanisms of these super-minds. I don't really have a huge amount of depth. I'm just like, this is very cool and it's fun to think about all of these things really.

## Implications for Alignment

So, I told you it's going to be speculative, and it's getting speculative. Let me try to bring this back together. What do we think about this for alignment? If we're humans, obviously maybe the super-mind isn't so great. What do we want to do about it? What can we do about it?

Obviously, if it's a singleton, we just got to make sure the singleton is aligned. We all agree on that. But if we have many AIs, what do we do?

I don't really have good answers here. I wish I did. These are my preliminary thoughts.

### Population Statistics

One thing is: if we have AI emerging from a population, maybe just the statistics of this population are important. They probably are. We should make the statistics of this population good. We should make as many AIs as aligned as possible as we can.

Obviously, there will be some misaligned AIs. Some people will go out crazy and create paper-clippers for fun. But at the same point, if there's a whole world of non-paper-clippers, they have very strong incentives to band together and stop the paper-clipper. The coalition politics will work in our favor at this point. In general, creating alignment and creating more aligned AIs is probably good in general.

### Overlapping Values

The other thing is we can achieve different degrees of alignment as long as the values of the alignment are overlapping. We think of alignment as a zero-one property—it's either aligned or it's not aligned. But in practice, people will probably align AIs to different things. People themselves have different values. We somehow manage to make it work out mostly.

Likely it will be similar with the AIs, assuming there's lots of overlap in the things that they're aligned to. Maybe the combined strength of these things will actually be sufficiently aligned in general. The intersection of all the different alignments will probably be good. We should just try in general—we can experiment a lot with different alignments as long as the intersection is somewhat decent for humans, which if humans succeed at alignment at all, it probably is.

### Integrating Humans

The other thing is maybe we would just want to integrate humans into this. Right now, we have the AI doing their weird mind stuff and humans are kind of sitting on the sidelines. We can't communicate this fast. We have to talk. We have to use language. Maybe we should stop that. Maybe we should figure out ways for humans to get better integrated into this AI society.

The kind of obvious way is we've got to improve our BCI technology. We've got to figure out ways that humans can have the same affordances as AIs with respect to their minds. How can we communicate human thoughts directly? Humans have their own unsupervised learning embedding space. It's somewhat similar to AI embedding spaces because of just natural representation convergence. We can directly integrate humans with this AI mind, with this AI economy, assuming we can actually figure out how to directly interface with people's brains, which is going to happen. That's going to be super interesting.

It's not just a world of AIs doing the AI thing and us just sitting here. We will also be, hopefully, deeply involved in this world.

### Political Philosophy Questions

Then there's also a super-interesting question really of political philosophy: suppose we're in this multi-mind setting—what does the game theory of cooperation look like? What are the values that are broadly appealing to all minds sufficient to encourage them to join some coalition together, and what do these values look like?

Is it—I discussed liberalism several times. Is there some kind of mind liberalism that exists which is some equilibrium solution here? Can we think of Rawlsian-style veil of ignorance? This is another solution to how multi-agent systems should cooperate and distribute resources. Are we going to have some weird convex combination of utility functions? Andrew Critch had a nice paper on this where it's like we can convexly combine utility functions together. This is cool. This basically results in the concept of equity. Some people have more power and more equity in the mind values than others.

Is this going to happen? Is this good? Is this bad? There's lots of interesting questions here.

That's basically the end. Thanks!


1.
I don't mean this in a sort of pejorative 'power relations/inequality' way, I mean in the 'structured like a tree' way, where there aren't overlaps or cross-links between subcommunities.

