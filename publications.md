---
layout: page
title: Publications
subtitle: Selected research programs and full publication list
permalink: /publications/
---

My research has moved between computational neuroscience, biologically plausible learning algorithms, and large-scale foundation models. Broadly I have been attempting to understand how intelligence works in both brains and machines, and specifically understanding questions such as how learning and inference can be implemented in local and asynchronous biological systems, how associative memories work, the underlying fundamental of exploration in RL, and most recently how architectures, data, and training systems come together to build highly capable foundation models across a wide range of modalities. 

[Google Scholar](https://scholar.google.com/citations?hl=en&oi=ao&user=3GGkFTkAAAAJ) · [GitHub](https://github.com/BerenMillidge) · [Full publication list](#all-publications)

---

# Selected research

The sections below collect a small number of papers into the main research programs I have worked on. They are intended as an entry point; the complete reverse-chronological publication list is below.

## Foundation models: architectures, data, and scaling

Since cofounding Zyphra, my work has increasingly focused on building and leading research across foundation-model architectures, data, training systems, as well as trying to attack fundamental questions of continual learning and long term memory. A recurring question is which constraints actually limit scaling, and how changes to architecture, data, or training dynamics can open new scaling axes.

### Selected papers

**[ZAYA1-8B Technical Report](https://arxiv.org/abs/2605.05365)** (2026)   
Robert Washbourne, Rishi Iyer, Tomas Figliolia, Henry Zheng, Ryan Lorig-Roach, Sungyeon Yang, Pritish Yuvraj, Quentin Anthony, Yury Tokpanov, Xiao Yang, Ganesh Nanduru, Stephen Ebert, Praneeth Medepalli, Skyler Szot, Srivatsan Rajagopal, Alex Ong, Bhavana Mehta, **Beren Millidge**

This technical report presents our state-of-the-art 8B LLM foundation model that we trained in-house end-to-end including pretraining, midtraining, RL. Uses a novel in-house architecture we developed (CCA and Zaya router). Out performs all contemporary models of its size and is competitive with substantially larger models including then-frontier models in certain mathematics and coding tasks. 

**[Training Foundation Models on a Full-Stack AMD Platform: Compute, Networking, and System Design](https://arxiv.org/abs/2511.17127)** (2025)  
Quentin Anthony, Yury Tokpanov, Skyler Szot, Srivatsan Rajagopal, Praneeth Medepalli, Rishi Iyer, Vasu Shyam, Anna Golubeva, Ansh Chaurasia, Xiao Yang, Tomas Figliolia, Robert Washbourne, Drew Thorstensen, Amartey Pearson, Zack Grossbart, Jason van Patten, Emad Barsoum, Zhenyu Gu, Yao Fu, **Beren Millidge**

A detailed systems paper on our novel full-stack AMD pretraining approach. We are the first to enable large-scale LLM pretraiing on an end-to-end AMD stack of MI300x GPUs and AMD Pollara networking. 

**[Scaling Adaptive Depth with Norm-Agnostic Residual Networks](https://arxiv.org/abs/2606.16112)** (2026)  
Tomás Figliolia, **Beren Millidge**

We introduce norm-agnostic residual streams, a novel method to prevent diminishment of marginal capacity growth with depth which exists in current models. 

**[Can Scale Save Us From Plasticity Loss in Large Language Models?](https://arxiv.org/abs/2606.24752)** (2026)  
J. Fernando Hernandez-Garcia, Tomás Figliolia, **Beren Millidge**

Here we study whether plasticity loss persists in modern language models and how its onset changes with scale. This thus connects continual learning questions to the the current contemporary LLM regime. 

**[The Zamba2 Suite](https://arxiv.org/abs/2411.15242)** (2024)  
Paolo Glorioso, Quentin Anthony, Yury Tokpanov, Anna Golubeva, Vasudev Shyam, James Whittington, Jonathan Pilault, **Beren Millidge**

Introduces the Zamba2 SSM–Transformer hybrid architecture, combining a Mamba backbone with shared attention to improve efficiency while retaining strong language-model performance. We trained then-SOTA LLMs in the 7B, 3B, and 1B size bracket. 

**[Zyda-2: a 5 Trillion Token High-Quality Dataset](https://arxiv.org/abs/2411.06068)** (2024)  
Yury Tokpanov, Paolo Glorioso, Quentin Anthony, **Beren Millidge**

An example of the data side of the foundation-model program. We constructed and open-sourced a trillion-token-scale pretraining dataset which outperformed comparable pretraining sets of the time, as well as released the full dataset processing, filtering, and deduplication infrastructure. 

## Predictive coding and local learning

Much of my PhD and postdoctoral work asked whether powerful learning algorithms such as backpropagation can emerge from local distributed dynamics, and whether predictive coding provides a useful general framework for inference and learning in biological and artificial networks.

### Selected papers

**[Predictive Coding Approximates Backprop along Arbitrary Computation Graphs](https://arxiv.org/abs/2006.04182)** (2020; later published in *Neural Computation*)  
**Beren Millidge**, Alexander Tschantz, Christopher L. Buckley

We were the first to demonstrate that local learning algorithms such as predictive coding can approximate backpropagation on arbitrary computation graphs, demonstrating a potential route for backprop-like algorithms to be implemented in neural circuitry. 

**[Inferring neural activity before plasticity as a foundation for learning beyond backpropagation](https://www.nature.com/articles/s41593-023-01514-1)** (2024 *Nature Neuroscience*)
Yuhang Song, **Beren Millidge**, Tommaso Salvatori, Thomas Lukasiewicz, Zhenghua Xu & Rafal Bogacz 

We developed *prospective configuration* a novel learning algorithm building upon predictive coding and show that it outperforms backpropagation on online and continual learning tasks. 

**[A Theoretical Framework for Inference and Learning in Predictive Coding Networks](https://arxiv.org/abs/2207.12316)** (2022; ICLR 2023)  
**Beren Millidge**, Yuhang Song, Tommaso Salvatori, Thomas Lukasiewicz, Rafal Bogacz

We developed a general framework for understanding how predictive coding networks differ from backpropagation trained networks, and how predictive coding relates to Gauss-Newton, Target-Propagation and other learning algorithms.

**[Backpropagation at the Infinitesimal Inference Limit of Energy-Based Models](https://arxiv.org/abs/2206.02629)** (2022; ICLR 2023)  
**Beren Millidge**, Yuhang Song, Tommaso Salvatori, Thomas Lukasiewicz, Rafal Bogacz

We developed a mathematical framework through which we can understand essentially the entire literature of biological learning algorithms approximating backprop through a unifying abstraction of the infinitesimal inference limit. 

**[Hybrid Predictive Coding: Inferring, Fast and Slow](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011280)** (PNAS 2023)  
Alexander Tschantz*, **Beren Millidge***, Anil Seth, Christopher Buckley

We combined iterative predictive-coding inference with amortized inference, thus linking biologically motivated local computation with learned feedforward inference. 


## Control, active inference, and value learning

An earlier strand of my work studied control and reinforcement learning through the lens of probabilistic inference. I was particularly interested in where exploration terms come from, the relationship between iterative planning and amortized policies, and how agents can flexibly represent and revalue multiple rewards.

### Selected papers

**[Whence the Expected Free Energy?](https://arxiv.org/abs/2004.08128)** (2020; *Neural Computation*, 2021)  
**Beren Millidge**, Alexander Tschantz, Christopher Buckley

We analyzed the mathematical origin of expected free energy and the relationship between active-inference objectives and information-seeking exploration.

**[Reward Bases: A Simple Mechanism for Adaptive Acquisition of Multiple Reward Types](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012580)** (PLOS Computational Biology 2024)  
**Beren Millidge**, Yuhang Song, Armin Lak, Mark E. Walton, Rafal Bogacz

Here, we developed a mechanism for rapidly recombining learned reward components as motivational state changes, enabling 'zero-shot' transfer of value functions and behaviour to novel reward functions depending on physiological state. 

**[Deep Active Inference as Variational Policy Gradients](https://www.sciencedirect.com/science/article/pii/S0022249620300298)**   (2020; Journal of Mathematical Psychology)
**Beren Millidge**

Here, I was the first to 'scale up' active inference to contemporary RL environments and agent scales, demonstrating that active-inference-inspired agents outperformed standard policy gradient and Q-learning approaches in deep RL tasks. 


**[Understanding the Origins of Information-Seeking Exploration in Probabilistic Objectives for Control](https://arxiv.org/abs/2103.06859)** (2021)  
**Beren Millidge**, Alexander Tschantz, Anil Seth, Christopher Buckley

Here we derive the original of information-seeking objectives in reinforcement learning as deriving from divergence minimizing rather than reward maximizing functionals.



## Associative memory and representations

A smaller but recurring line of my research studies associative memory as a general computational primitive and its connections to representation learning, attention, and graph-based retrieval.

### Selected papers

**[Universal Hopfield Networks: A General Framework for Single-Shot Associative Memory Models](https://arxiv.org/abs/2202.04557)** (2022; ICML 2022)  
**Beren Millidge**, Tommaso Salvatori, Yuhang Song, Thomas Lukasiewicz, Rafal Bogacz

Here, we unified a large literature of existing disparate associative memory models and placed them all into a common framework by decomposing retrieval into similarity, separation, and projection operations. We demonstrated that this unification allows the immediate implementation of novel similarity and separation functions that outperformed existing assoiative memory methods. 

**Associative Memories in the Feature Space** (2023; ECAI 2023)  
Tommaso Salvatori, **Beren Millidge**, Yuhang Song, Rafal Bogacz

We demonstrate that associative memory models can be made substantially more efficient and performant if the associative operation is performed upon a learnt latent feature space rather than in raw input/output space. We demonstrate that such latent associative memories outperform then-current methods operating on the output space. 

**[Hybrid Associative Memories](https://arxiv.org/abs/2603.22325)** (2026)  
Leon Lufkin, Tomas Figliolia, **Beren Millidge**, Kamesh Krishnamurthy

We developed a novel hybrid memory which combined SSMs and full attention in a novel way by using the SSM to process the majority of the seauence while only passing to attention the tokens which are *surprising* to the SSM. We demonstrated that this outperformed existing SSM hybrid methods. 

**[Mixture-of-PageRanks: Replacing Long-Context with Real-Time, Sparse GraphRAG](https://arxiv.org/abs/2412.06078)** (2024)  
Nicholas Alonso, **Beren Millidge**

We developed a novel page-rank inspired RAG mechanism which allowed perfect and SOTA performance on challenging retrieval benchmarks on context lengths of up to a billion tokens while running in real-time entirely on the CPU


# All publications

Below is a complete publication record presented in reverse chronological order. I will try to keep this list up to date, however an always up to date list can be found at my [Google Scholar](https://scholar.google.com/citations?user=3GGkFTkAAAAJ&hl=en&oi=ao).

## 2026

**PUFFER: Incremental Fuzzy Deduplication for Continuously Evolving Corpora** (2026) <br /> Xiao Yang, Erik Edward Aldape, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2608.28622)

**ZUNA1.1: A more flexible EEG foundation model for Denoising and Super-resolution** (2026) <br /> Christopher Warner, Jonas Mago, JR Huml, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2607.27308)

**ZONOS2 Technical Report** (2026) <br /> Gabriel Clark, Sofian Mejjoute, Mohamed Osman, George Close, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2606.24320)

**Can Scale Save Us From Plasticity Loss in Large Language Models?** (2026) <br /> J. Fernando Hernandez-Garcia, Tomás Figliolia, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2606.24752)

**Scaling Adaptive Depth with Norm-Agnostic Residual Networks** (2026) <br /> Tomás Figliolia, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2606.16112)

**Zamba2-VL Technical Report** (2026) <br /> Hassan Shapourian, Kasra Hejazi, Olabode M. Sule, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2606.00390)

**ZAYA1-VL-8B Technical Report** (2026) <br /> Hassan Shapourian, Kasra Hejazi, Olabode M. Sule, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2605.08560)

**ZAYA1-8B Technical Report** (2026) <br /> Robert Washbourne, Rishi Iyer, Tomas Figliolia, Henry Zheng, Ryan Lorig-Roach, Sungyeon Yang, Pritish Yuvraj, Quentin Anthony, Yury Tokpanov, Xiao Yang, Ganesh Nanduru, Stephen Ebert, Praneeth Medepalli, Skyler Szot, Srivatsan Rajagopal, Alex Ong, Bhavana Mehta, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2605.05365)

**Hybrid Associative Memories** (2026) <br /> Leon Lufkin, Tomas Figliolia, **Beren Millidge**, Kamesh Krishnamurthy <br /> [paper](https://arxiv.org/abs/2603.22325)

**ZUNA: Flexible EEG Superresolution with Position-Aware Diffusion Autoencoders** (2026) <br /> Christopher Warner, Jonas Mago, JR Huml, Mohamed Osman, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2602.18478) \| [code](https://github.com/Zyphra/zuna)

**Online Vector Quantized Attention** (2026) <br /> Nick Alonso, Tomas Figliolia, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2602.03922)

## 2025

**Equivalence of Personalized PageRank and Successor Representations** (2025) <br /> **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2512.24722)

**Generalizing E-prop to Deep Networks** (2025) <br /> **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2512.24506)

**Training Foundation Models on a Full-Stack AMD Platform: Compute, Networking, and System Design** (2025) <br /> Quentin Anthony, Yury Tokpanov, Skyler Szot, Srivatsan Rajagopal, Praneeth Medepalli,Anna Golubeva, Vasu Shyam, Robert Washbourne, Rishi Iyer, Ansh Chaurasia, Tomas Figliolia, Xiao Yang, Drew Thorstensen, Amartey Pearson, Zack Grossbart,Jason van Patten, Emad Barsoum, Zhenyu Gu, Yao Fu, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2511.17127v1)

**Compressed Convolutional Attention: Efficient Attention in a Compressed Latent Space** (2025) <br /> Tomas Figliolia, Nicholas Alonso, Rishi Iyer, Quentin Anthony, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2510.04476)

## 2024

**Mixture-of-PageRanks: Replacing Long-Context with Real-Time, Sparse GraphRAG** (2024) <br /> Nicholas Alonso, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2412.06078)

**The Zamba2 Suite: Technical Report** (2024) <br /> Paolo Glorioso, Quentin Anthony, Yury Tokpanov, Anna Golubeva, Vasudev Shyam, James Whittington, Jonathan Pilault, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2411.15242) \| [code](https://github.com/Zyphra/Zamba2)

**Reward Bases: A simple mechanism for adaptive acquisition of multiple reward types** (2024) <br /> **Beren Millidge**, Yuhang Song, Armin Lak, Mark E Walton, Rafal Bogacz <br /> [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012580) \| [code](https://github.com/YuhangSong/reward-bases)

**Zyda-2: a 5 Trillion Token High-Quality Dataset** (2024) <br /> Yury Tokpanov, Paolo Glorioso, Quentin Anthony, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2411.06068)

**Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters** (2024) <br /> Vasudev Shyam, Jonathan Pilault, Emily Shepperd, Quentin Anthony, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2408.04093) \| [code](https://github.com/Zyphra/tree_attention)

**Zyda: A 1.3 T Dataset for Open Language Modeling** (2024) <br /> Yury Tokpanov\*, **Beren Millidge**\*, Paolo Glorioso, Jonathan Pilault, Adam Ibrahim, James Whittington, Quentin Anthony <br /> [paper](https://arxiv.org/abs/2406.01981) \| [code](https://github.com/Zyphra/Zyda_processing)

**Toward Conversational Agents with Context and Time Sensitive Long-term Memory** (2024) <br /> Nicholas Alonso, Tomas Figliolia, Anthony Ndirango, **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2406.00057)

**Zamba: A Compact SSM Hybrid Model** (2024) <br /> Paolo Glorioso\*, Quentin Anthony\*, Yury Tokpanov\*, James Whittington, Jonathan Pilault, Adam Ibrahim, **Beren Millidge**\* <br /> [paper](https://arxiv.org/abs/2405.16712) \| [code](https://github.com/Zyphra/Zamba-torch)

**A Review of Neuroscience-Inspired Machine Learning** (2024) <br /> Alexander Ororbia, Ankur Mali, Adam Kohan, **Beren Millidge**, Tommaso Salvatori <br /> [paper](https://arxiv.org/abs/2403.18929)

**Natural Induction: Spontaneous adaptive organisation without natural selection** (2024) <br /> Christopher L Buckley, Tim Lewens, Michael Levin, **Beren Millidge**, Alec Tschantz, Richard A Watson <br /> [paper](https://www.biorxiv.org/content/10.1101/2024.02.28.582499v1.abstract)

**BlackMamba: Mixture of Experts for State Space Models** (2024) <br /> Quentin Anthony\*, Yury Tokpanov\*, Paolo Glorioso\*, **Beren Millidge**\* <br /> [paper](https://arxiv.org/abs/2402.01771) \| [code](https://github.com/Zyphra/BlackMamba)

## 2023

**Collective Behaviour from Surprise Minimization** (2023) <br /> Conor Heins, **Beren Millidge**, Lancelot Da Costa, Richard Mann, Karl Friston, Iain Couzin <br /> [paper](https://arxiv.org/abs/2307.14804)

**Predictive Coding Networks for Temporal Prediction** (2023) <br /> **Beren Millidge**, Mufeng Tang, Mahyar Osanlouy, Rafal Bogacz <br /> [paper](https://www.biorxiv.org/content/biorxiv/early/2023/05/16/2023.05.15.540906.full.pdf)

**Exploring Action-Centric Representations through the Lens of Rate-Distortion Theory** (2023) <br /> Miguel De Llanza Varona, Christopher Buckley, **Beren Millidge** <br /> [paper](https://openreview.net/pdf?id=C-UXIjnKox)

**Causal Inference via Predictive Coding** (2023) <br /> Tommaso Salvatori, Luca Pinchetti, Amine M'Charrak, **Beren Millidge**, Thomas Lukasiewicz <br /> [paper](https://arxiv.org/abs/2306.15479)

**Associative Memories in the Feature Space** (2023) <br /> Tommaso Salvatori, **Beren Millidge**, Yuhang Song, Rafal Bogacz <br /> [paper](https://www.researchgate.net/publication/374324823_Associative_Memories_in_the_Feature_Space)

**From the free energy principle to a confederation of Bayesian mechanics. Reply to comments on" How particular is the physics of the free energy principle?"** (2023) <br /> Miguel Aguilera, **Beren Millidge**, Alexander Tschantz, Christopher Buckley <br />[paper](https://ui.adsabs.harvard.edu/abs/2023PhLRv..44..270A/abstract)

## 2022

**Generalized Predictive Coding: Bayesian Inference in Static and Dynamic models** (2022) <br /> Andre Ofner, **Beren Millidge**, Sebastian Stober <br /> [paper](https://openreview.net/forum?id=qaT_CByg1X5) 

**Recurrent predictive coding models for associative memory employing covariance learning** (2022) <br /> Mufeng Tang, Tommaso Salvatori, **Beren Millidge**, Yuhang Song, Thomas Lukasiewicz, Rafal Bogacz <br /> [paper](https://www.biorxiv.org/content/10.1101/2022.11.09.515747v1.abstract)

**Incremental Predictive Coding: A Parallel and Fully Automatic Learning Algorithm** (2022) <br /> Tommaso Salvatori, Yuhang Song, **Beren Millidge**, Zhenghua Xu, Lei Sha, Cornelius Emde, Rafal Bogacz, Thomas Lukasiewicz <br /> [paper](https://arxiv.org/abs/2212.00720)

**Predictive Coding Beyond Gaussian Distributions** (2022) <br /> Luca Pinchetti, Tommaso Salvatori, Yordan Yordanov, **Beren Millidge**, Yuhang Song, Thomas Lukasiewicz <br /> [paper](https://arxiv.org/abs/2211.03481)

**Capsule Networks as Generative Models** (2022) <br /> Alex B Kiefer\*, **Beren Millidge\***, Alexander Tschantz\*, Christopher Buckley  <br /> [paper](https://arxiv.org/pdf/2209.02567.pdf) \| [Alex's code](https://github.com/exilefaker/capsnet-experiments),  [my code](https://github.com/BerenMillidge/Sparse_Routing)

**Preventing Deterioration of Classification Accuracy in Predictive Coding Networks** (2022) <br /> Paul F Kinghorn, **Beren Millidge**, Christopher L Buckley <br /> [paper](https://arxiv.org/pdf/2208.07114.pdf)

**A Theoretical Framework for Inference and Learning in Predictive Coding Networks** (2022) <br /> **Beren Millidge**, Yuhang Song, Tommaso Salvatori, Thomas Lukasiewicz, Rafal Bogacz <br /> [paper](https://arxiv.org/abs/2207.12316) \| [code](https://github.com/BerenMillidge/theoretical_framework_predictive_coding)

**Successor Representation Active Inference** (2022) <br /> **Beren  Millidge**, Christopher L Buckley <br /> [paper](https://arxiv.org/abs/2207.09897) \| [code](https://github.com/BerenMillidge/Active_Inference_Successor_Representations)

**A Theoretical Framework for Inference Learning** (2022) <br /> Nick Alonso, **Beren Millidge**, Jeff Krichmar, Emre Neftci <br /> [paper](https://arxiv.org/pdf/2206.00164.pdf) \| [code](https://github.com/nalonso2/ILTheory)

**Backpropagation at the Infinitesimal Inference Limit of Energy-Based Models: Unifying Predictive Coding, Equilibrium Propagation, and Contrastive Hebbian Learning** (2022) <br /> **Beren Millidge**, Yuhang Song, Tommaso Salvatori, Thomas Lukasiewicz, Rafal Bogacz <br /> [paper](https://arxiv.org/pdf/2206.02629.pdf) \| [code](https://github.com/BerenMillidge/infinitesimal_inference_limit)

**On Bayesian Mechanis: A physics of and by beliefs** (2022) <br /> Maxwell JD Ramstead, Dalton AR Sakthivadivel, Conor Heins, Magnus Koudahl, **Beren Millidge**, Lancelot Da Costa, Brennan Klein, Karl J Friston <br /> [paper](https://arxiv.org/pdf/2205.11543.pdf)

**Inferring Neural Activity Before Plasticity: A Foundation for Learning Beyond Backpropagation** (2022) <br /> Yuhang Song, **Beren Millidge**, Tommaso Salvatori,Thomas Lukasiewicz, Zhengua Xu, Rafal Bogacz <br /> [paper](https://www.biorxiv.org/content/biorxiv/early/2022/05/18/2022.05.17.492325.full.pdf) \| [code](https://github.com/YuhangSong/A-New-Perspective)

**Reward Bases: Instantaneous Reward Revaluation with Temporal Difference Learning** (2022) <br /> **Beren Millidge**, Mark Walton, Rafal Bogacz <br /> [paper](https://www.biorxiv.org/content/10.1101/2022.04.14.488361v1) \| [code](https://github.com/BerenMillidge/Reward_Bases)

**Hybrid Predictive Coding: Inferring, Fast and Slow** (2022) <br /> Alexander Tschantz\*, **Beren Millidge\***, Anil Seth, Christopher Buckley. <br /> [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011280) \| [code](https://github.com/alec-tschantz/pybrid) 

**Predictive Coding: Towards a Future of Deep Learning beyond Backpropagation?** (2022) <br /> **Beren Millidge\***, Tommaso Salvatori\*, Yuhang Song, Rafal Bogacz, Thomas Lukasiewicz <br /> [paper](https://arxiv.org/pdf/2202.09467.pdf)

**Universal Hopfield Networks: A General Framework for Single-Shot Associative Memory Models** (2022) <br /> **Beren Millidge**, Tommaso Salvatori, Yuhang Song, Thomas Lukasiewicz, Rafal Bogacz <br /> [paper](https://arxiv.org/abs/2202.04557) \| [code](https://github.com/BerenMillidge/Theory_Associative_Memory)

**Learning on Arbitrary Graph Topologies via Predictive Coding** (2022) <br /> Tommaso Salvatori, Luca Pinchetti, **Beren Millidge**, Yuhang Song, Rafal Bogacz, Thomas Lukasiewicz <br />[paper](https://arxiv.org/pdf/2201.13180.pdf)

**pymdp: A Python library for active inference in discrete state spaces** (2022) <br /> Conor Heins, **Beren Millidge**, Daphne Demekas, Brennan Klein, Karl Friston, Iain Couzin, Alexander Tschantz <br /> [paper](https://arxiv.org/abs/2201.03904) \| [code](https://github.com/infer-actively/pymdp)

## 2021

**Active Inference in Robotics and Artificial Agents: Survey and Challenges** (2021) <br /> Pablo Lanillos, Cristian Meo, Corrado Pezzato, Ajith Anil Meera, Mohamed Baioumy, Wataru Ohata, Alexander Tschantz, **Beren Millidge**, Martijn Wisse, Christopher L. Buckley, Jun Tani <br /> [paper](https://arxiv.org/abs/2112.01871)

**Habitual and Reflective Control in Hierarchical Predictive Coding** (2021) <br /> Paul F Kinghorn, **Beren Millidge**, Christopher L Buckley <br /> [paper](https://arxiv.org/pdf/2109.00866.pdf)

**A Mathematical Walkthrough and Discussion of the Free Energy Principle** (2021) <br /> **Beren Millidge**, Anil Seth, Christopher Buckley <br /> [paper](https://arxiv.org/abs/2108.13343)

**Predictive Coding: A Theoretical and Experimental Review** (2021) <br /> **Beren Millidge**, Anil Seth, Christoper Buckley <br /> [paper](https://arxiv.org/abs/2107.12979)

**Applications of the Free Energy Principle to Machine Learning and Neuroscience** (2021) <br /> **Beren Millidge** <br /> [paper](https://arxiv.org/abs/2107.00140) \| [code](https://github.com/BerenMillidge/PhD_Thesis)

**Online Reinforcement Learning with Sparse Rewards through an Active Inference Capsule** (2021) <br /> Alejandro Daniel Noel, Charel van Hoof, **Beren Millidge** <br/> [paper](https://arxiv.org/pdf/2106.02390.pdf) \| [code](https://github.com/adanielnoel/Active-Inference-Capsule)

**Towards a Mathematical Theory of Abstraction** (2021) <br />**Beren Millidge** <br /> [paper](https://arxiv.org/pdf/2106.01826.pdf)

**How Particular is the Physics of the Free Energy Principle** (2021) <br /> Miguel Aguilera, **Beren Millidge**, Alexander Tschantz, Christopher Buckley <br /> [paper](https://arxiv.org/pdf/2105.11203.pdf)

**Understanding the Origins of Information-Seeking Exploration in Probabilistic Objectives for Control** (2021) <br /> **Beren Millidge**, Alexander Tschantz, Anil Seth, Christopher Buckley <br/> [paper](https://arxiv.org/pdf/2103.06859.pdf) \| [code](https://github.com/BerenMillidge/origins_information_seeking_exploration)

**Neural Kalman Filtering** (2021) <br /> **Beren Millidge**, Alexander Tschantz, Anil Seth, Christopher Buckley <br/> [paper](https://arxiv.org/pdf/2102.10021.pdf) \| [code](https://github.com/BerenMillidge/NeuralKalmanFiltering)

## 2020

**Sophisticated Active Inference: Simulating Anticipatory Affective Dynamics of Imagining Future Events** (2020) <br /> Casper Hesp, Alexander Tschantz, **Beren Millidge**, Maxwell Ramstead, Karl Friston, Ryan Smith <br /> [paper](https://www.researchgate.net/profile/Casper_Hesp2/publication/344750468_Sophisticated_Affective_Inference_Simulating_Anticipatory_Affective_Dynamics_of_Imagining_Future_Events/links/5f8d9cf7458515b7cf8b7aff/Sophisticated-Affective-Inference-Simulating-Anticipatory-Affective-Dynamics-of-Imagining-Future-Events.pdf) <br />
Published in *IWAI IEEE workshop on Active Inference*

**Investigating the Scalability and Biological-Plausibility of the Activation Relaxation Algorithm** (2020) <br/> **Beren Millidge**, Alexander Tschantz, Anil Seth, Christopher L Buckley <br/> [paper](https://arxiv.org/abs/2010.06219.pdf) \| [code](https://github.com/BerenMillidge/Dynamical-Activation-Relaxation) <br />
Published in *NeurIPS 2020 workshop on Backpropagation in the Brain*

**Relaxing the Constraints on Predictive Coding Models** (2020) <br/> **Beren Millidge**, Alexander Tschantz, Anil Seth, Christopher L Buckley <br/> [paper](https://arxiv.org/pdf/2010.01047.pdf) \| [code](https://github.com/BerenMillidge/RelaxedPredictiveCoding)

**Activation Relaxation: A Local Dynamical Approximation to Backprop in the Brain** (2020)    <br/> **Beren Millidge**, Alexander Tschantz, Anil Seth, Christopher L Buckley <br/> [paper](https://arxiv.org/abs/2009.05359) \|  [code](https://github.com/BerenMillidge/ActivationRelaxation)

**The Acquisition of Culturally Patterned Attention Styles under Active Inference** (2020) <br/> Axel Constant, Alexander Tschantz, **Beren Millidge**, Felipe Criado-Boado, Luis M Martinez, Johannes Müller, Andy Clark. <br/> [paper](https://psyarxiv.com/rchaf/) \|  [code](https://github.com/BerenMillidge/MaterialCulture)

**Control as Hybrid Inference** (2020) <br/> Alexander Tschantz, **Beren Millidge**, Anil Seth, Christopher Buckley <br/> [paper](https://arxiv.org/pdf/2007.05838.pdf) <br/>
Published in *ICML 2020* workshop.

**On the Relationship between Control as Inference and Active Inference** (2020) <br/> **Beren Millidge**, Alexander Tschantz, Anil Seth, Christopher Buckley <br/> [paper](https://arxiv.org/pdf/2006.12964.pdf) <br/>
Published in *IWAI IEEE workshop on Active Inference*

**Reinforcement Learning as Iterative and Amortised Inference** (2020) <br/> **Beren Millidge\***, Alexander Tschantz*, Christopher Buckley <br/> [paper](https://arxiv.org/abs/2006.10524)

**Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs** (2020) <br/> **Beren Millidge**, Alexander Tschantz, Christopher Buckley <br/>
[paper](https://arxiv.org/abs/2006.04182) \|  [code](https://github.com/BerenMillidge/PredictiveCodingBackprop)


**Curious Inferences:** *Reply to Sun & Firestone on the Dark Room Problem* (2020) <br/>
Anil Seth, **Beren Millidge**, Christopher Buckley, Alexander Tschantz <br/>
[paper](https://psyarxiv.com/w8y9p/) <br/>
Published in *Trends in Cognitive Science*  

**Whence the Expected Free Energy** (2020) <br/>
**Beren Millidge**, Alexander Tschantz, Christopher Buckley <br/>
[paper](https://arxiv.org/abs/2004.08128) <br/>
Published in *Neural Computation*

**Reinforcement Learning Through Active Inference** (2020) <br/>
Alexander Tschantz\*, **Beren Millidge\***, Anil Seth, Christopher Buckley <br/> 
[paper](https://arxiv.org/abs/2002.12636)  \|  [code](https://github.com/alec-tschantz/rl-inference)  <br/>
Published in *Bridging AI and Cognitive Science (ICLR 2020) workshop*

## 2019

**Deep Active Inference as Variational Policy Gradients** (2019) <br/>
**Beren Millidge** <br/>
Published in *Journal of Mathematical Psychology*  <br/>
[paper](https://arxiv.org/pdf/1907.03876.pdf) \| [code](https://github.com/BerenMillidge/DeepActiveInference)

**Combining Active Inference and Hierarchical Predictive Coding: A Tutorial Introduction and Case-Study** (2019) <br/>
**Beren Millidge** <br/>
[paper](https://psyarxiv.com/kf6wc/) \|  [code](https://github.com/BerenMillidge/Combining-Active-Inference-Paper-Code)

**Implementing Predictive Processing and Active Inference: Preliminary Steps and Results** (2019) <br/>
**Beren Millidge** <br/>
[paper](https://psyarxiv.com/4hb58/) \| [code](https://github.com/BerenMillidge/Implementing_Predictive_Processing)

**Vocal imitation can create acoustic attractors to guide mothers to pups in a crowded colony of Mexican free-tailed bats: A case-study of computational modelling in behavioural biology** (2019) <br/>
Richard Shillcock, **Beren Millidge**, Andrea Ravignani <br/>
[paper](https://psyarxiv.com/9y652/) \| [code](https://github.com/BerenMillidge/Vocal_Learning)

**Fixational Eye Movements: Data Augmentation for the Brain?** (2019) <br/>
**Beren Millidge** <br/>
[paper](https://www.researchgate.net/publication/331909056_Fixational_Eye_Movements_Data_Augmentation_for_the_Brain)  \|  [code](https://github.com/BerenMillidge/RetinalStabilisation)   

**Exploring infant vocal imitation in Tadarida brasiliensis mexicana** (2019) <br>
Richard Shillcock, **Beren Millidge**, Andrea Ravignani (2019) <br/>
Published in *Neurobiology of Speech and Language* <br/>
[paper](https://elibrary.ru/item.asp?id=39139444)

## 2018

**A Predictive Processing Account of Bottom-Up Visual Saliency Using Cross-Predicting Autoencoders** (2018) <br/>
**Beren Millidge**, Richard Shillcock <br/>
[paper](https://psyarxiv.com/csmeb/)  \|  [code](https://github.com/BerenMillidge/Saliency)

