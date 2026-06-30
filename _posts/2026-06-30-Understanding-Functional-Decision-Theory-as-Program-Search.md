---
layout: post
title: Understanding Functional Decision Theory as Program Search
---

**Epistemic note**: *Probably already known or obvious, but this treatment seems clearer to me than the standard treatment in e.g. [the FDT paper](https://arxiv.org/abs/1710.05060)*. 

The other day I went down a deep lesswrong rabbit-hole and ended up going through all the decision theory literature, especially relating to [Functional Decision Theory](https://arxiv.org/abs/1710.05060) (FDT). For those who joined lesswrong later, questions of decision theory and specifically the 'best' or 'optimal' decision theory for agents to use was an important question in the early, pre-deep-learning days of lesswrong, and is at the heart of many agent foundations questions and agendas. The idea here is that if we can figure out the best decision theory, we can start to reason concretely about what optimal agents should do, and from there try to reach for some method of mathemtically provable alignment which is stable under reflection. 

While the idea of expected utility maximization is clear intuitively, exactly how to define and implement it concretely -- i.e. how to define a 'decision theory' -- is surprisingly subtle and contested. 'Classical' decision theories such as [Evidential Decision Theory](https://plato.stanford.edu/entries/decision-theory/) (EDT) and [Causal Decision Theory](https://plato.stanford.edu/archives/fall2022/entries/decision-causal/) (CDT) give controversial and seemingly incorrect decisions in problems with logical dependence between choices and outcomes such as Newcomb's problem. To address this, Eliezer and Nate Soares proposed [Functional Decision Theory](https://arxiv.org/abs/1710.05060) (FDT) which gives better answers in various Newcomb-like problems. The guiding principle here is stated as: *'the normative principle for action is to treat one’s decision as the output of a fixed mathematical function that answers the question, “Which output of this very function would yield the best outcome?”'*

Personally, when I originally was deeply involved in the early lesswrong circles I never really understood what exactly this meant, and in general I found that FDT definitions and discussion pretty opaque. The standard description of FDT given in the paper and elsewhere routes through a strange definition of counterfactuals and tries very closely to tie FDT to notions in mathematical logic and modal theory. 

What I want to do here is provide what is, in my opinion, a much clearer definition of FDT that is grounded in standard computational primitives. Specifically, my definition of FDT is based around program search. Namely, that given a computable representation of the environment, *FDT performs a search across all possible policy programs to pick the one that exits with the maximum utility*. 

The real advantage here, which is what I think reduces the confusion, is that by thinking of things in terms of Turing-complete programs, we immediately gain a very clear notation and mechanism which makes things such as *logical dependence* extremely obvious, and almost trivial, while trying to represent these notions in mathematical logic or standard probabilistic graph notation ends up being very convoluted. Given the representation of environemnts and policies as programs, FDT becomes almost trivial and we can instantly see where the other decision theories are failing. 

### Classical Decision Theories

Let's start at the beginning. A standard staple of economics, game theory, and many other fields is that agents have a utility function and that, given this utility functions, they should select actions that maximize their utility. Classically, we assume that an agent has a utility function which satisfies the VNM axioms and that assigns a scalar value to every possible outcome. Where there is stochasticity, or epistemic uncertainty, the agent should instead select actions that lead to the highest *expected* utility under the posterior distribution. Ignoring issues of computational tractability in practice, conceptually this approach seems clear.

The naive application of this idea leads to Evidentiary Decision Theory (EDT). EDT proposes that the agent should simply be an optimal Bayesian and choose the actions that are associated with the highest utility. Specifically, the agent should select the action that, when conditioned upon, leads to the highest utility states on average. Let $x$ refer to a state of the world and $a$ as a possible action. Let $U(x)$ denote the utility of a state. Throughout, for simplicity, we will only handle discrete state and action spaces. The EDT procedure can then be formalized as follows,

$$
\begin{aligned}
    a^* = \operatorname*{argmax}_a \sum_x p(x \mid a) U(x)
\end{aligned}
$$

Where the $\sum_x p(x \mid a)U(x)$ implements the expected value over the utility of all states weighted by their likelihood conditioned on a specific action. 

However, a subtle problem arises since the EDT operates only at the level of Bayesian inference over conditional distributions. Without a notion of the true causal graph underlying the world, it can be susceptible to errors when causality and correlation are decoupled. This is because there is a fundamental difference between an agent observing that actions and states are correlated in some abstract sense and directly taking those actions itself. Actions and states can be correlated in a misleading way if there is a third mediator variable that is directly correlated to both actions and states but which is not present when the agent tries to take the actions for itself. One classic example where EDT fails is in the [Smoking Lesion problem](https://www.jstor.org/stable/2026547) where the existence of a mediator that causes both smoking and cancer misleads the agent into making an incorrect decision. 

This is a classic issue in probabilistic inference in general which was ignored and caused confusion for a surprisingly long time. Luckily, the work of [Sewall Wright](https://en.wikipedia.org/wiki/Sewall_Wright), and [Judea Pearl](https://en.wikipedia.org/wiki/Judea_Pearl) etc has furnished us with a decent theory of *causal inference*, not simply correlational inference. Causal inference proposes that we have some underlying causal graph of the environment which defines the linkages between various variables. Given this graph, we can then infer the impact of various 'interventions' upon variables within this graph. This lets us get correct predictions where the naive 'correlational' inference methods fail. 

If we integrate the Pearlian framework into our decision theory, we get Causal Decision Theory (CDT). CDT proposes that the optimal actions should be chosen by creating a causal graph model of the environment, then modeling its actions as interventions via Pearl's $\text{do}(\cdot)$ operator, and then selecting the actions where the expected utility of the states given an intervention on the action is highest. CDT successfully solves the kinds of problems where correlation and causation come apart where EDT fails. CDT can be mathematically formulated as,

$$
\begin{aligned}
    a^* = \operatorname*{argmax}_a \sum_x p(x \mid \text{do}(a)) U(x)
\end{aligned}
$$

Where behind $\text{do}(x)$ lies a causal graph of the world that underlies the conditional distribution $p(x /mid a)$, as is standard in the causality literature. The core idea here is that we should model an agent's own actions as *sui generis*. I.e. the agent is an unconstrained actor in the world which can sever otherwise existing causal dependencies between variables by exerting its own will upon the value of these variables -- i.e. by performing an *intervention*. 

However, while CDT succeeds at solving the kinds of causality problem that mislead EDT, it makes controversial, and arguably incorrect decisions in cases involving more subtle kinds of dependence than direct causal dependence. The most obvious kind of non-causal dependence is *logical dependence*. This means that agents, or some aspect of the environment acts in highly correlated ways even if they are not causally linked. Some classic ways to get logical dependence is via identical agents -- i.e. agents reasoning about what their perfect clone or twin will do -- or perfect predictors such as *Omega* from Newcomb's problem. FDT attempts to extend CDT by solving this additional logical dependence problem. 

### Functional Decision Theory

A classic example of this kind of problem is [Newcomb's problem](https://en.wikipedia.org/wiki/Newcomb%27s_problem). The setup for this problem is as follows,

**Newcomb's Problem**: *An agent is contacted by a perfect predictor "Omega". Omega offers a choice of two boxes: "A" that contains &#36;1,000, and "B" that contains either &#36;1,000,000 or &#36;0. Omega is known to be able to predict an agent's choice with near-100% accuracy. Omega claims to have placed &#36;1,000,000 in box B if and only if it predicted that the agent would leave box A behind. Omega has made its prediction and filled both boxes prior to contacting the agent, and so box "B" either already contains the &#36;1,000,000 or &#36;0. Should the agent take both boxes ("two-boxing") or only box B ("one-boxing")?*

In this case, since by the time the agent has been presented with the boxes by Omega, box B already contains the money or is empty, there is thus no causal link between the agent's action and Omega's previous decision to select whether to put money in box B or not. Hence, reasons a CDT agent, it should two-box since whatever is in box B is already decided, so it may as well take the extra "free" &#36;1,000. Of course, the counterargument is that Omega, by definition, can predict this course of action and hence places &#36;0 in box B. By contrast, the EDT agent sees that the probability of receiving the &#36;1,000,000 by choosing to one-box is higher -- i.e., $p(1{,}000{,}000 \mid \text{one-box}) > p(1{,}000{,}000 \mid \text{two-box})$ and so one-boxes.

A similar situation occurs in the twin prisoner's dilemma problem. Here we have an agent which is tasked to play a single shot game of prisoner's dilemma against an identical twin of itself. If the agent chooses to cooperate or defect, so will its twin. In this problem, the CDT agent reasons that when it is facing the agent, it is always better to defect since by defecting it will gain utility over its twin vs cooperating. However, in practice this leads the CDT agents to always enter a defect-defect state, which is suboptimal. Similarly in this situation, the EDT agent will make the decision to cooperate since it observes that cooperation is correlated with higher utility in the end. Here we observe the EDT agent making seemingly more 'rational' decisions than the CDT agent, at least ones that give greater utility in expectation. However, given EDTs problems with classic causality issues, it is clear that returning to EDT is not the solution. 

Dissatisfaction with these aspects of CDT brings us to Functional Decision Theory (FDT). FDT changes the decision objective again to state that an agent should instead choose the *decision process* which counterfactually leads to the best outcomes, rather than intervening on actions directly. Mathematically, FDT introduces the $\text{FDT}(\cdot)$ operator which takes in a set of information about the states, causal graph, and *logical dependencies* about the world and then outputs a decision procedure. FDT then chooses an action by intervening on the outcome of its decision procedure to ask what action would produce the highest utility. FDT can be formalized in our scheme as,

$$
\begin{aligned}
    a^* = \operatorname*{argmax}_a \sum_x p(x \mid \text{do}(\text{FDT}(x...) = a)) U(x)
\end{aligned}
$$

Where $x...$ represents the states $x$ augmented with some notion of causal graph and other 'logical relations' however defined. The FDT reasoning in Newcomb's paradox is that rather focusing in on the moment of choice, we should look at the outcomes of the decision procedure as a whole. Overall one-boxers clearly do better in Newcomb's paradox, since they receive the $\$1M$ while CDT two-boxers do not and only receive \$1000. Hence, one should choose to one-box, even if at the precise moment of decision it seems 'irrational' to leave the other box sitting there untouched when there is nothing Omega can do to stop you from taking it. Nearly identical reasoning plays out in the prisoner's dilemma against a copy scenario. Clearly the cooperators do better overall, and hence it is rational to cooperate even if at the moment of decision defecting is strictly superior. FDT, unlike CDT, also does not give in to acausal blackmail for very similar reasons. 

While FDT solves problems of decision theory in situations of logical dependence which befuddle CDT agents, there remain several details to work out. First, the definition of the $FDT(\cdot)$ operator is somewhat fuzzy as well as the definition of logical or 'subjunctive' dependence and how to determine whether different variables are dependent in this way. Secondly, the FDT authors claim that there remains an unsolved problem of logical counterfactuals, since the FDT's $\text{do}(\cdot)$ operator requires the ability to evaluate logical counterfactuals such as 'what if this deterministic decision procedure outputted a different truth value'.

# Formulation as Program Search

The goal here is to provide a simpler and clearer reformulation of FDT and the decision theory problem more generally. The ultimate objective is to make FDT seem trivial and indeed the most obvious and naive decision theory given the surrounding formulation. 

Specifically, we want to reframe the decision theory setting from one of probabilistic/causal inference to a very concrete setting of 'program optimization' where both the environment and the policy of the agent are defined as computable, Turing-complete programs, rather than probability distributions or causal graphs.

Given this definition of the environment, a decision procedure becomes a search over programs that maximize expected utility, where the utility is the return value of the environment. Under these definitions, FDT becomes extremely simple to define as the naive unbounded optimization over program space, while CDT and EDT become subsets of FDT that impose restrictions on the kinds of programs they can consider, namely causal graphs and Bayesian graphical models respectively. This approach neatly takes into account all kinds of dependence -- logical, causal, and probabilistic -- in a unified and simple framework, since all are inherent within the correct programmatic description of the environment and the policy. 

Mathematically, our framework defines two functions *Env* and *Policy*. *Env* accepts as input a policy and returns a scalar utility value which is computed by the interaction of the environment with the policy (*Env* is thus, mathemtically, a functional*). *Policy* receives as input a sequence of observations from the environment and outputs the relevant actions. Both *Env* and *Policy* can internally implement arbitrary Turing-complete programs. Mathematically, we can formulate our FDT-style decision procedure very simply as just,

$$
\begin{aligned}
    \texttt{Policy}^* 
    = \operatorname*{argmax}_{\texttt{Policy}}
       \texttt{Env}(\texttt{Policy})
\end{aligned}
$$

Note that this definition is substantially simpler than CDT and the original formulation of FDT. We do not require the definition of any special operator like the 'do' operator in CDT and the 'FDT' operator. Instead, we follow the standard utility maximization approach but simply change the maximization domain to be over policy programs rather than specific sets of actions. 

We can express our decision theory in a more natural form as code as well. For simplicity, let's express all code in a Pythonic-style language. Our FDT decision rule is just,

```python
def compute_optimal_policy(env, all_possible_policies):
    policies_list = [p for p in all_possible_policies...]
    utilities = [env(p) for p in policies_list]
    best_utility_idx = argmax(utilities)
    best_policy = policies_list[best_utility_idx]
    return best_policy
```

Obviously, doing this approach naively is usually computationally intractable since the set of all possible policies is often infinite. We don't attempt to directly solve this here since we are just dealing with theoretical decision theory rather than the practical implementation of anything.

Althougoh slightly subtle, the most interesting thing here is our formulation of the *environment*. We formulate the environment as a program which accepts a policy (i.e., a decision procedure) as a parametrized input and returns a scalar utility value. Note that this formulation of both the environment and the policy as *computational programs* renders notions of self-reference and logical 'subjunctive' dependence, substantially clearer and less confusing. These notions become mapped to assignment of a value to multiple variables, multiple invocations of a function and potentially recursion which are well understood and clear in computer science. Programs are extremely good media for handling logical dependence and self-reference without causing any confusion or opacity. 

Secondly, under our definitions, a policy is just a program which receives observations from the environment and emits actions back to the environment. For every observation it can emit an action. It is not required to utilize observations and can just trivially return the same actions for any observation sequence -- i.e., an ['updateless'](https://www.lesswrong.com/posts/de3xjFaACCAk6imzv/towards-a-new-decision-theory) [policy](https://arxiv.org/abs/2306.00175). The policy itself can have arbitrary, but finite, programmatic complexity. For a real-world agent, it would likely involve substantial inference and optimization budget for creating and tracking an internal state and performing planning. We ignore all these details here since we are concerned with decision theory in the abstract. 

### Reformulating and Solving Classic Decision-Theory Scenarios

To illustrate the theory, let us consider several classic problems in decision theory which highlight well-known deficiencies of CDT which FDT resolves. First let us consider the [identical twin prisoner's dilemma problem](https://www.lesswrong.com/w/psychological-twin-prisoner-s-dilemma). We can model the environment for this as a program as follows:

```python
def twin_PD_environment(policy):
    your_move = policy()
    opponent_move = policy() # since your twin plays your policy
    
    if your_move == 'cooperate' and opponent_move == 'cooperate':
        return 10
    if your_move == 'cooperate' and opponent_move == 'defect':
        return -20
    if your_move == 'defect' and opponent_move == 'cooperate':
        return 20
    if your_move == 'defect' and opponent_move == 'defect':
        return -5
```

Where the policy space is assumed to contain just two policies -- one which always returns "cooperate" and another which always returns "defect". If we then apply our decision procedure from above to this environment, we find that, as expected, the 'cooperate' policy achieves a utility of 10 and the 'defect' policy achieves a score of -5. So we should cooperate, as FDT stipulates. Note, however, that we did not need to do anything special to handle the intrinsic logical dependence between your move and your twin's move. Here, this is handled very naturally in the program for the environment itself by simply having your twin call your policy directly. This kind of logical dependence between variables is completely natural and entirely unremarkable in programming languages. 

Next, let us consider Newcomb's problem. We formulate the Newcomb's problem environment as follows,

```python
def Newcombs_problem_environment(policy):
    fill_box_A_with_1000_usd()
    Omega_prediction = policy()  # since Omega is a perfect predictor

    # Omega sets up the boxes
    if Omega_prediction == 'onebox':
        fill_box_B_with_1000000_usd()
    else:
        fill_box_B_with_0_usd()

    # Your policy executes
    your_decision = policy()

    # Utility is computed
    if your_decision == 'onebox':
        return utility_box_B()
    if your_decision == 'twobox':
        return utility_box_A() + utility_box_B()
```

Here, let's just define the policy space as containing two policies: the 'onebox' or the 'twobox' policies. We observe that the programmatic structure and logical dependence of Newcomb's problem is practically identical to the twin prisoner's dilemma. This is because a perfect predictor and your identical twin are secretly the same logical concept. Here we set Omega's prediction to be perfect since it literally runs the agent's policy. The same program formulation, however, can easily handle some degree of randomness or uncertainty in Omega's prediction. Our programmatic formulation also handles the causality problems that CDT solves since the causal graph is necessarily encoded as part of the program for the environment.

Our programmatic approach also provides insight into the differences between FDT and CDT. We can actually trace this down to a difference between the *definition of the environment* rather than the decision theory. Specifically, CDT assumes that environments can support arbitrary Pearlian interventions on the action nodes *during runtime* of the program, while FDTs notion of the environment does not support this but instead only allows the agent to change its 'a-priori' policy. To make this clearer, we define a program for a CDT-style Newcomb's problem.

```python
def Newcombs_problem_environment(policy):
    fill_box_A_with_1000_usd()
    Omega_prediction = policy() # Since Omega is a perfect predictor
    # Omega sets up the boxes
    if Omega_prediction == 'onebox':
        fill_box_B_with_1000000_usd()
    else:
        fill_box_B_with_0_usd()

    # Your decision is computed
    your_decision = policy()
    # Intervention!
    your_decision = intervention_choice() # Intervention overrides policy

    # Utility is computed
    if your_decision == 'onebox':
        return utility_box_B()
    if your_decision == 'twobox':
        return utility_box_A() + utility_box_B()
```

The CDT environment model therefore grants the agent the ability to arbitrarily interrupt and edit the 'flow' of the running program of the environment. This is the operation performed by the Pearlian $\text{do}(\cdot)$ operator, since it severs all causal connections to the intervened node. In effect, the intervention emerges from 'ex-nihilo' or some force outside the causal model and has the power to arbitrarily re-arrange things within the causal computation. If this is the case, then the CDT approach of two-boxing is rational since the intervention can override Omega's perfect prediction abilities.

From this view, Newcomb's-like paradoxes are not actually paradoxes of decision-theory, instead they are disagreements over the precise nature of the *environment* and the nature of possible interactions that an agent can have with the environment. Specifically, CDT assumes it is possible to define 'interventions' that can change both the causal graph and the 'logical graph' directly 'at runtime' in a way that is not detectable by Omega. FDT assumes that this is not possible and the only lever of action the agent has is selecting the policy, which Omega can predict.  

As an intuition-pump for this. Consider Newcomb's problem where the agent is running on some physically instantiated hardware. The program it is running commands one-boxing, which Omega, of course, observes and thus places the $\$1M$ in the box. However, by sheer cosmic fluke, at the exact moment of decision, a cosmic-ray hits the hardware running the agent causing a bit-flip that causes the agent to two-box, and let's assume that Omega could not have predicted the cosmic ray, then the agent walks away with both rewards. This unpredicted and unmodeled cosmic ray is essentially what a CDT 'intervention' is modeling. By its lights, if its definition of the environment is correct, then CDT is the correct decision theory. If true 'ex-nihilo' interventions to the causal state are possible and not modelled by Omega, then you can have your cake and eat it too by two-boxing in Newcomb's-like problems. If you cannot, and your decision-making can be modeled by Omega -- i.e., only changed at the 'policy selection' ('compile-time') level and not at the 'execution' ('runtime') level, then you should follow FDT and one-box. 

As an aside, it is also worth discussing the argument and critique [here](https://arxiv.org/abs/1507.01986) that an FDT agent performs poorly in the ultimatum-game-like 'tens game' against an unsophisticated opponent. The setup here is an FDT agent vs another player where each agent has to pick a number less than ten. If the sum of both numbers is less than ten such as '4' and '5', then each agent receives utility proportional to their number. If the sum is greater than ten both agents get nothing. If the opposing agents knows that the FDT agent is a perfect predictor, then they can go ahead and pick '8' with confidence knowing that the FDT agent will predict this and thus pick '1' which is the best option it can do. The worry is that how can an FDT agent be 'rational' if it can be so easily exploited by another agent.  

However, there is no paradox here. It is not that being a perfect predictor itself is bad, rather it is *your opponent knowing your policy* which is bad. This is true regardless of whether your policy is a perfect predictor or some other arbitrary policy. If the human opponent did not know the FDT agent's policy, then it would be optimal. However, in cases where the other agent is expected to be somewhat successful at modeling or guessing your policy, more complex policies are needed. This kind of recursive modelling of different strategies leads to randomized algorithms and sets of increasingly complex statistical policies [in practice](https://www.lesswrong.com/posts/AGZD62scqRaoM6p4n/rock-paper-scissors-is-not-solved-in-practice).

In these scenarios, credible precommitment to removing options from yourself and/or playing a naively suboptimal policy (vs naive agents) such as randomizing your actions is actually the superior policy. Examples include Chicken where directly removing your ability to swerve is beneficial and playing randomized actions in a game like Rock-Paper-Scissors. This is not a contradiction or paradox and can be modeled in this framework by having both the agent and the other agents in the environment explicitly model each other's policies, which can be easily represented within our Turing-complete program framework (since presumably this modelling is Turing complete and eventually halts otherwise there is no game). If we expand the policy set from just policies which hardcode a specific number in response to the opponent to policies which can e.g., randomize their selection, then the FDT agent can again perform very well even against an agent which knows that their opponent is an omniscient FDT agent.

In any case, hopefully this formulation of decision theory is simpler and clearer than the standard treatment, and makes FDT seem obvious in retrospect. This is true of many good ideas. The ultimate message I want to get across here is that FDT is not a strange or subtle idea but in fact the simplest and most obvious decision theory when formulated in the right manner. It is simply standard utility maximization when performed over policies and environments that are general programs, which is naturally the most general setting and likely the proper way to represent things in the first place. This should be expected in retrospect, because at the abstract level decision theory should not be complex. The basic idea of expected utility maximization is extremely simple; we just need to find the right frame to make it simple in.

There are a couple of interesting additional thoughts and ideas that follow from this. Firstly, the idea of representing environments/policies as programs is hardly new, but it is still somewhat rare. I think this is a very general and good move in the long term. Many ideas in e.g. [probabilistic and differentiable programming](https://en.wikipedia.org/wiki/Probabilistic_programming) build in exactly this direction by converting our standard probabilistic graphical models into programs and directly reasoning about them. General Pearl-style causality is also much easier to reason about in terms of programs than in terms of DAGs. I.e. a Pearlian intervention is simply overwriting a value in a program at runtime. AI makes this view easier to come to, since the ultimately the core insight is creating parametrizable differentiable programs which thus become learnable. 

It's also interesting that this provides a new perspective on the differences between the various decision theories and their respective limitations. We see here that FDT is actually the most general in that it makes the least assumptions about the structure of the environment, while other decision theories make specific assumptions that cause them to fail in cases where these assumptions do not hold. Specifically, EDT has a fundamentally impoverished representation of the world only through probability distributions over observation variables rather than the true causal graph (or program) underlying the world. The reason this is a problem is that there is a nonidentifiability problem wherein many potential causal graphs can generate the same probability distribution over observations, however they have different implications for future observations. EDT thus fails in any scenarios where correlation $\neq$ causation, many cases of which have been amply demonstrated in the [causal inference literature](https://www.cambridge.org/core/books/causality/B0046844FAE10CBF274D4ACBDAEB5F5B) and elsewhere. Where EDT succeeds in e.g., Newcomb-like problems, it succeeds 'by accident' since the logical relations which are not directly modeled by EDT also end up as correlational relations which are. However, it is straightforward to design Newcomb-like variants where EDT also fails, which demonstrates that EDT is not really solving the problem. Similarly, CDT can encode the causal graph but since it only encodes causal relationships and not logical relationships, CDT will fail in all cases where there is un-modelled logical dependence, which includes Newcomb's-like problems as a subcase. This is all somewhat obvious but seeing it all tie together like this is neat. 

One other interesting aspect and difference between CDT and FDT, however, is the question of how interventions are modeled in the environment. If 'true' interventions, which can supersede logical relationships such as perfect prediction and identity are allowed, then the CDT approach is correct. Mathematically, this occurs when the $\text{do}(\cdot)$ is allowed to sever logical connections as well as causal connections.  If however, interventions can only happen in the deciding of the policy, which thus preserve logical relations, then FDT is correct. Our new frame makes this distinction clear. 

Finally, and this should be obvious, but working out decision theory in this manner does not give us much in terms of actually making decisions in practice. For abstract decision theory, we simply focus on the decision rule and the formulation of the setting. We do not deal with any practical problems such as how to construct this programmatic model of the environment in the first place?; how to practically search the potentially infinite policy space, and so on.  These problems are common to all decision theories and are typically assumed away. However, we note that practical agents, beyond simply using a good decision theory, also need to find practical algorithms to approximate the state of the world and perform policy search in a computationally efficient and tractable manner.

Even at the abstract level, by formulating our environment and policies as arbitrary programs, our framework is vulnerable to standard uncomputability issues. For instance if, for some reason, our environment or policy involved determining whether an arbitrary program will halt within finite time, the environment would become incomputable.  For now, we do not concern ourselves with such issues and we believe that these kinds of problems are unlikely to arise for practical agents. Another, more serious, possibility is for agents to get stuck in an infinite regress of simulating one another. This can occur in many seemingly straightforward game-theoretic scenarios. In practice, this can be resolved by allowing stochastic policies or, more exotically, through [reflective oracles](https://arxiv.org/abs/1508.04145).
