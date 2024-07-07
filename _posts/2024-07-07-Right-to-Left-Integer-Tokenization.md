---
layout: post
title: Right to Left (R2L) Integer Tokenization
---

*This is a guest post by Max Buckley, a software engineer at Google and fellow AI researcher[^1].*

By some twist of fate, this blog has become the chronicle of the evolution of integer tokenization. In an [earlier post in February of 2023](https://www.beren.io/2023-02-04-Integer-tokenization-is-insane/), it was discussed how older models: GPT2 and GPT3 tokenized integers by naively and directly applying Byte Pair Encoding (BPE) to numbers. It was shown how bizarre and arbitrary the results of such a process were. The GPT2 and 3 tokenizer assigned a large number of integers their own tokens, while other numbers were split arbitrarily. Because of the BPE tokenization scheme, the model assigned unique tokens to commonly occuring numbers such as small numbers, as well as recent years (1930-2020), and common longer values 10000 as their own token.An implication of this number partitioning scheme is that the model is forced to learn the associations and rules of arithmetic again and again with varying amounts of data and simply memorize a large number of arithmetic operations involving unique tokens.

In [a more recent post in May](https://www.beren.io/2024-05-11-Integer-tokenization-is-now-much-less-insane/), we studied the integer tokenization of more recent models and found that it was much less insane. This is because  most more recent models (GPT3.5 onwards) have moved away from pure BPE and converged to one of two strategies:

1.) Tokenize individual digits

2.) Tokenize sequences of up to three digits

Both of these strategies ensure that there is a fixed number of integer tokens visible to the model (10 or 1110 respectively) and the model learns the relationships between those number tokens and no others in a consistent and coherent way, enabling it to learn simple serial algorithms for performing arithmetic operations instead of a vast and arbitrary lookup table.

The former strategy of tokenizing single digits appears to be more popular in smaller models (Mistral, Meta’s Llama 1 and 2, Google’s Gemma) and is not the focus of this post.

The latter strategy, tokenizing up to three digits is more widely adopted by frontier models. It is in use by OpenAI’s more recent models (GPT3.5 and later), Anthropic’s recent Claude models (3 and above), Google’s Gemini models, and Meta’s Llama 3 models amongst others. In that same May 2024 post, we also stated that “it seems that the problem has been solved”, and while it is true that the problem of numbers being consistently tokenized was solved, the problem of LLM performance on simple arithmetic tasks still leaves much to be desired.

# The problem of arithmetic

Historically, most models struggle with simple arithmetic when manipulating large numbers, the degree of these problems depends on a range of factors, the size of the numbers, the size of the model, the number of examples provided (few shots), the model architecture etc. But a major common factor is the tokenization of numbers.

A very promising research paper was released in February of 2024, The authors, Aaditya Singh and DJ Strouse showed that forcing models to use right to left (R2L) number tokenization rather than the current industry standard left to right (L2R) number tokenization lead to huge increases in performance on simple arithmetic tasks (addition being the example shown). In their paper they tested GPT 3.5 and GPT 4 showing the value of moving to R2L by adding separators to numbers (typically commas, but any separator can do). They showed increases in accuracy from 75.6% to 97.8% for GPT 3.5 and 84.4% to 98.9% for GPT4. This R2L direction of separation is likely because integers are typically represented from most significant digit on the left to least significant digit on the right. Hence many serial algorithms for arithmetic operations such as addition actually operate right to left. 

# What is R2L tokenization?

By default, tokenizers tokenize text, including numbers left to right (L2R). So given the string “1000” and the three digit strategy mentioned above,  this would be tokenized into tokens greedily. The example “1000” becomes [“100”, “0”]. A right to left tokenization of this number would instead yield [“1”, “000”]. This latter R2L representation probably looks more familiar as that is how we commonly represent numbers, breaking them into chunks of size three from right to left using separators. The R2L representation thus prevents the tokenization boundary of the number shifting when there is a carry.

# What is happening now?

In March of 2024 Anthropic released their newest generation of Claude models, Claude 3. Anthropic does not share many details of their models, including not sharing their tokenizer, so did not publicly state either their huge improvement in arithmetic performance or the changes they had made to their number tokenization.

However in April of 2024 the CEO of Anthropic, Dario Amodei was making some impressive claims about Claude's performance at arithmetic on a podcast (43:00 onwards) and prompted the community to investigate. As it turns out, one of the improvements made in the Claude 3 series is the change to R2L number tokenization. With this change in place Claude 3 and beyond have far superior arithmetic capabilities. Even their smallest and least capable Claude model (Haiku) outperforms the flagship models from its competitors (OpenAI, Google, Meta).

We ran a range of experiments inspired by the work of Aaditya Singh and DJ Strouse. Effectively we used the same system prompt and 8 shot examples and then randomly sampled two (three token) integers between 1000000 and 999999999 and had the various models add those numbers and compared the model’s response to what the actual addition should generate. We did n=300 replications for each model in both default and with added commas to force this R2L tokenization behavior.

# Results

The results are ordered by their default performance (lowest to highest).

| Supplier | Model | Default Performance | Default Strategy | Forced R2L 3 digit (With Commas) | Improvement |
| -------- | ----- | ------------------- | ---------------- | ------------------------------- | ----------- |
| Anthropic | claude-2.1 | 28.3 | BPE | 100 | 71.7 |
| OpenAI | gpt-3.5-turbo | 50.3 | L2R 3 digit | 97 | 46.7 |
| Meta | Meta-Llama-3-70B-Instruct | 68 | L2R 3 digit | 98 | 30 |
| OpenAI | gpt-4 | 90.6 | L2R 3 digit | 100 | 9.4 |
| Google | gemini-1.5-pro | 92 | L2R 3 digit | 100 | 8 |
| OpenAI | gpt-4o | 94 | L2R 3 digit | 98.3 | 4.3 |
| Anthropic | claude-3-haiku-20240307 | 99 | R2L 3 digit | - | - |
| Anthropic | claude-3-5-sonnet-20240620 | 100 | R2L 3 digit | - | - |

Expanding to much larger random numbers: 10^17 to  (10^(20) -1).

| Supplier | Model | Default Performance | Default Strategy | Forced R2L 3 digit (With Commas) | Improvement |
| -------- | ----- | ------------------- | ---------------- | ------------------------------- | ----------- |
| Anthropic | claude-2.1 | 0.3 | BPE | 72.7 | 72.4 |
| OpenAI | gpt-3.5-turbo | 35.3 | L2R 3 digit | 85 | 49.7 |
| OpenAI | gpt-4o | 89.7 | L2R 3 digit | 96.3 | 6.6 |
| Anthropic | claude-3-haiku-20240307 | 94.3 | R2L 3 digit | - | - |
| Anthropic | claude-3-5-sonnet-20240620 | 100 | R2R 3 digit | - | - |

# Discussion

The improvements in performance were visible for all models and were especially large for smaller models (GPT 3.5 Turbo) and those using BPE (Claude 2.1). It seems evident that this is a significantly valuable optimization and likely can be further expanded or explored by the research community. We also showed that making the numbers even larger caused the accuracy to fall even further, with GPT4o going from 94% to 89.7%, GPT3.5 Turbo going from 50.3% to 35.3%, and Claude 2.1 falling from 28.3% to 0.3%. While the R2L performance held up much better. Claude 3.5 Sonnet held constant at 100%.

If model developers can patch this optimization on top of existing tokenizers to mimic this change into their own models at inference time only, i.e. can be utilized without retraining. If Tiktokenizer, SentencePiece or similar can add a boolean flag to tokenize numbers R2L and it can be switched on with little to no side effects at inference time then that would be a great win for all model providers. If not then it can at least be used to tokenize the training data before training the next generation of models begin training.

Moreover a benchmark of this type should be adopted as a standard test for all models. We focused on the addition of 7 to 9 and 17 to 19 digit numbers, but a standard benchmark could easily also explore other operators (subtraction, multiplication, etc) and other number sizes and is relatively trivial to implement and simple to verify.

| Model | Strategy |
| ----- | -------- |
| GPT-3 (2020) | pure BPE |
| GPT-3.5 (2022) | L2R chunks of 3 digits |
| GPT-4 (2023) | L2R chunks of 3 digits |
| Claude v2.1 (2023) | pure BPE |
| Llama 1 & 2 (2023) | single digit |
| Mistral (2023) | single digit |
| Gemini 1.5 Pro (2024) | L2R chunks of 3 digits |
| GPT-4o (2024) | L2R chunks of 3 digits |
| Llama 3 (2024) | L2R chunks of 3 digits |
| Claude 3 (2024) | R2L chunks of 3 digits |

# Speculation

Given the recent paper, the analysis above, and especially Claude 3’s state of the art arithmetic performance relative to its peers, we expect all future models to adopt such a R2L strategy. Moreover if this can be patched onto existing tokenizers and used at inference time with no retraining, we expect some of these improvements to materialize very quickly, significantly ameliorating one of the weaknesses of modern LLMs.

# Footnote on the Anthropic Tokenizer

The Anthropic Claude 3 tokenizer is not public, but insights can be gleaned by using either their workbench or from their API and its responses.

Both their workbench and API (max_tokens) allow you to specify a limit to the number of returned tokens. This when combined with a prompt like: 

```
Copy "1000" and nothing else.
```

- With a limit of 1 token returns nothing
- With a limit of 2 tokens returns “1”
- With a limit of 3 tokens returns “1000”

This tells us that Anthropic adds a special mystery token before numbers, it also tells us that the numbers are tokenized R2L. If you do the same analysis with a competitor tokenizer you would get 100 as the first number token and 0 as the second. This can be verified in Tiktokenizer.

What is this mystery token that precedes numbers? We are not fully sure yet. It does not occur if you ask for a non number character like “a” or “!”. It also doesn’t occur if the number character is preceded by many other characters. e.g. a letter character. 

- “7” is two tokens
- "a7" is two tokens, 
- "7a" is three tokens


[^1]: Contributions: Max wrote a draft on this post and did the experiments, Beren provided editorial review.
