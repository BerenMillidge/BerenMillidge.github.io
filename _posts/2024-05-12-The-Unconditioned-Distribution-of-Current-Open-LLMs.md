---
layout: post
title: The Unconditioned Distribution of Current Open LLMs
---

Last year, I wrote a [quick post](https://www.beren.io/2023-02-26-Fingerprinting-LLMs-with-unconditioned-distribution/) investigating the 'unconditioned' distribution of LLMs in the OpenAI API, where the 'unconditioned distribution' is simply the distribution of LLM outputs following the empty string -- or beginning of sequence token. My intuition here was that this gives some idea of what the LLM thinks is 'intrinsically most likely' and is likely to be the kind of thing seen most often in training. For the OpenAI API, with the 3.5 series of models, this resulted in some interesting conclusions, such as that several supposedly different models had the same unconditioned distribution, likely implying that they were either trained on the same dataset or were all derived from the same source base model. 

Now, with the explosion of open-source models coming available, I thought I would repeat this exercise to see if there is anything we can learn from the most recent open-source models. One of the things I am interested in is the extent to which we can tell that new models with closed datasets are 'annealed' ala [miniCPM](https://arxiv.org/abs/2404.06395) or else include significant amount of instruct and finetuning data inside the pretraining as this appears to be one of the significant ways by which the performance of open models has been improved over the past year. There is definitely some suggestive evidence of this. For instance, taking the new [Llama3 8B](https://llama.meta.com/llama3/) base model, the unconditioned distribution is a math question/answer pair.

> <|begin_of_text|>Question:
> Let p = 0.2 + -0.3. Let x = 0.1 + p. Let g = 0.1 + x. Which is the closest to g?  (a) -0.5  (b) 0.1  (c) -0.4
> Answer:
> b<|end_of_text|>

This kind of data is very unlikely to show up in standard web corpora, thus implying that the significant performance boost of Llama3 models vs e.g. Llama2 may have come both from the 15T tokens and also using large amounts of instruct and 'eval-like' data during the pretraining process.

The outputs of [phi3](https://arxiv.org/abs/2404.14219), another model which is known to be trained on significant amounts of synthetic data also reveals this as the most likely sequence is what looks to be part of a coding question and response by a chatGPT-like model with an alpaca-like format.

> I've tried using `sed` but I'm not getting the desired output.
> ### response
> To replace all occurrences of `foo` with `bar` in a file using `sed`, you can use the following command:
> ```bash
> sed -i 's/foo/bar/g' filename
> ```
> This command uses `sed` to search for the pattern `foo` and replace it with `bar` glob

Another model, which we know was annealed is [Olmo-v1.7](https://huggingface.co/allenai/OLMo-1.7-7B). They explicitly discuss annealing on Flan datasets. Sure enough, the unconditioned output of Olmo-v1.7 is a stack-exchange like coding question

> Q: How to get the value of a variable in a function in a different file? I have a function in a file called "functions.js" and I want to get the value of a variable in  that function in another file called "index.js".
> I tried to use the "require" function but it didn't work.
> Here is the code:
> functions.js
> var myVar = "Hello";

It is possible that [Llama2](https://arxiv.org/abs/2307.09288) followed a similar prescription, or at least upweighted math and code questions in their training data since the unconditioned distribution of Llama2, has a similar form of coding questions:

> <s> Tags: c#, asp.net, asp.net-mvc, asp.net-mvc-4, asp.net-mvc-5
> Question: How to get the value of a checkbox in a view in ASP.NET MVC
> I have a view with a checkbox and a button.
> \begin{code}
> <input type="checkbox" name="chk" value="1" />
> <input type="submit" value="Submit" />
> \end


Mistral instruct (crucially not mistral base) also appears to follow a similar pattern with a very similarly formatted math question:

> <s> Q: Let n = q + d. Let q be (-52)/14 + 6/(-21). Let d = 29 + -25. Which is biger: -3/5 or n?
> A: n</s>

However, this method is not perfect or other frontier models have reached their performance by another route. For instance, [Mistral base](https://arxiv.org/abs/2310.06825), does not appear to output instruct-style question/answer pairs as its unconditioned distribution:

> <s> # The 10 Best Things to Do in Tulsa, Oklahoma
> Tulsa, Oklahoma is a city that is full of life and culture. There are many things to do in Tulsa, Oklahoma, and it can be hard to decide what to do first. Here are the 10 best things to do in Tulsa, Oklahoma.
> \## 1. Visit the Philbrook Museum of Art
> The Philbrook Museum of Art is a

But instead general descriptive text more reminiscent of a base model. Given that Llama3 needed to resort to both 15T tokens and clear instruct data during pretraining to match the performance of Mistral, it seems likely that Mistral originally used a similar approach. However, for whatever reason it does not show up in the unconditioned distribution. Or, potentially, Mistral used a different approach to reach that level of performance which is still not matched by Llama and Gemma.

[Google's Gemma model](https://arxiv.org/abs/2403.08295) also does not conform to the pattern and instead seems overtrained and simply spits out endless \<\bos> tokens as its unconditioned distribution, which is highly unusual. Either the version I downloaded from huggingface is slightly broken somehow or else during training there were (accidentally?) long strings of only the bos token. This is interesting since in the technical report they made an elliptical mention of an annealing-like two-phase training process.

However, this method does not appear to be foolproof. For instance models that are explicitly claimed to be annealed do not necessarily reveal this in their unconditioned distribution. For instance,[JetMoE](https://github.com/myshell-ai/JetMoE), which explicitly discusses annealing following miniCPM has a fairly standard-looking C-code repo as it's most likely string:

> <s> <reponame>johnny-morrice/nativefs
> #include <stdio.h>
> #include <stdlib.h>
> #include <string.h>
> #include <unistd.h>
> #include <errno.h>
> #include <sys/types.h>
> #include <sys/stat.h>
> #include <fcntl.h>
> #include <sys/mman.h>

Similarly, the well-known [Mistral OpenHermes](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B) finetune also appears to have a wiki article about how to make a birdhouse as it's most likely string, despite the fact that OpenHermes is known to be trained on a large, open dataset of instruction, code, and math data.

> <|im_start|> 
> # How to Make a DIY Birdhouse
> 
> Making a DIY birdhouse is a fun and easy project that can be completed in just a few hours. It’s a great way to provide a home for birds in your backyard and watch them come and go. Here’s how to make a DIY birdhouse:
> 
> Materials:
> 
> - 1×6 cedar board (8 feet long)

Overall, while looking at the unconditioned distribution provides a very cheap and easy first pass at trying to understand the train set a model uses, and does provide a glimpse at that, it definitely is far from an infallible tool for reverse-engineering what kind of training data the model was trained on. However, here we simply explore the most likely completions. Ultimately, since these models are generative models of the dataset, and good models have to be well-calibrated, extracting the shape of the training data should ultimately just be as simple as sampling many generations from the model starting from the empty string at increasingly high temperature. This should provide a fairly good measure of the typical kind of data the model saw during training, at least over short sequences before the model's own mistakes in generation push it out of distribution. I suspect also that by looking at the entropy of different generations, and how crystallized the model's responses are we can infer things about how overtrained the model is, potentially infer the methods used to train it, and ultimately maybe extract segments of the training data.
