---
layout: post
title: Current neural networks are not overparametrized
---

Occasionally I hear people say or believe that NNs are overparametrized and base their intuitions off of this idea. Certainly there is a small literature in academia around phenomena like double descent which do implicitly assume an overparametrized network. 

However, while overparametrized inference and generalization is certainly a valid regime to study neural networks in, it is important realize that essentially no ‘real world’ neural networks are in this regime today. Certainly neural networks can be very large, but this is in no way a guarantee of overparametrization. Overparametrization specifically refers to the *ratio* between the parameters and the dataset size, specifically if there are more parameters than data-points.

I think this idea emerges from historical ideas around the original Kaplan scaling laws. These laws heavily focused on the parameter size of the model and mostly ignored the dataset size, leading to extremely large models to be trained on relatively small datasets. The Chinchilla scaling laws, which as far as I know still remain the primary public scaling law study, focus significantly more on data and argue for in-tandem scaling of dataset size and model size, leading to optimal neural networks always remaining underparametrized.

Recent advances have further tilted the field towards ‘overtraining’ on even more data than Chinchilla would recommend, due to the increasing importance of inference economics and the fundamental costs involved in inferencing large models. This has led to models becoming increasingly under-parametrised relative to dataset size over time, with probably the 2020-2022 era being that of greatest overparametrization of models. This change has been very significant. For instance the original GPT3 was 175B parameters and trained on approximately 300B tokens for a ratio of slightly less than 2 tokens per parameter. This ratio is getting pretty close to overparametrized. Recent models such as Llama-3-8b are 8 billion parameters and trained on 15T tokens for just under 2000 tokens per parameter — a roughly 1000x decrease in overparametrization since 2020.

This does not mean that studying overparametrized networks is useless, however, it means that our models are firmly in the standard and well-understood classical statistical modelling regime where all the classic concepts of bias and variance and regularisation and overfitting still apply.
