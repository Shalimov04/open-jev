# BTZSC — `open-jev-base` (prereg docs/prereg/bench-v1.md)

17-clean mean macro-F1 **0.6049**, 95 % CI [0.6015, 0.6079] (2000 paired bootstrap draws over texts, seed 0)

**Verdict (§4): Competitive with the published field at 140M**

field median (35 models, 17 datasets) 0.5752; 22-dataset mean (all, incl. the 5 trained-on sources) 0.6011

| dataset | K | texts | macro-F1 | 95 % CI | acc | field median | ECE | status |
|---|---|---|---|---|---|---|---|---|
| yahootopics | 10 | 50000 | 0.5496 | [0.5453, 0.5538] | 0.5605 | 0.483 | 0.395 | clean |
| wikitoxic_insult | 2 | 8427 | 0.6825 | [0.6726, 0.6931] | 0.6848 | 0.595 | 0.101 | clean |
| wikitoxic_obscene | 2 | 8691 | 0.8458 | [0.8376, 0.8534] | 0.8464 | 0.683 | 0.278 | clean |
| wikitoxic_threat | 2 | 5211 | 0.4108 | [0.3994, 0.4221] | 0.5331 | 0.290 | 0.082 | clean |
| wikitoxic_toxicaggregated | 2 | 10000 | 0.7490 | [0.7405, 0.7576] | 0.7525 | 0.655 | 0.171 | clean |
| amazonpolarity | 2 | 10000 | 0.9091 | [0.9031, 0.9146] | 0.9091 | 0.912 | 0.041 | clean |
| imdb | 2 | 10000 | 0.8573 | [0.8502, 0.8639] | 0.8584 | 0.896 | 0.036 | clean |
| appreviews | 2 | 4000 | 0.8759 | [0.8660, 0.8856] | 0.8760 | 0.908 | 0.022 | clean |
| yelpreviews | 2 | 10000 | 0.9160 | [0.9104, 0.9216] | 0.9162 | 0.951 | 0.042 | clean |
| rottentomatoes | 2 | 1066 | 0.7631 | [0.7364, 0.7884] | 0.7645 | 0.814 | 0.076 | clean |
| empathetic | 32 | 2542 | 0.1909 | [0.1746, 0.2049] | 0.2042 | 0.318 | 0.134 | clean |
| biasframes_intent | 2 | 3648 | 0.5905 | [0.5736, 0.6066] | 0.5951 | 0.565 | 0.039 | clean |
| biasframes_offensive | 2 | 3838 | 0.6230 | [0.6079, 0.6380] | 0.6313 | 0.543 | 0.060 | clean |
| biasframes_sex | 2 | 4404 | 0.5052 | [0.4900, 0.5202] | 0.7386 | 0.304 | 0.172 | clean |
| trueteacher | 2 | 8955 | 0.4702 | [0.4600, 0.4812] | 0.4709 | 0.409 | 0.120 | clean |
| manifesto | 56 | 17018 | 0.0410 | [0.0347, 0.0480] | 0.0571 | 0.162 | 0.017 | clean |
| capsotu | 21 | 3355 | 0.3042 | [0.2811, 0.3245] | 0.3449 | 0.462 | 0.211 | clean |
| agnews | 4 | 7600 | 0.8356 | — | 0.8346 | 0.740 | 0.195 | trained on this source |
| banking77 | 72 | 3080 | 0.5675 | — | 0.5594 | 0.392 | 0.488 | trained on this source |
| emotiondair | 6 | 2000 | 0.4871 | — | 0.6095 | 0.421 | 0.164 | trained on this source |
| financialphrasebank | 3 | 690 | 0.7280 | — | 0.8000 | 0.496 | 0.081 | trained on this source |
| massive | 59 | 2974 | 0.3214 | — | 0.3302 | 0.411 | 0.254 | trained on this source |

## Published field, recomputed on the same 17 datasets

| model | params | 17-mean | 22-mean |
|---|---|---|---|
| Qwen3-Reranker-8B | 8B | 0.7316 | 0.7224 |
| Mistral-Nemo-Instruct-2407 | 12.2B | 0.7060 | 0.6697 |
| Qwen3-8B | 8.2B | 0.6970 | 0.6649 |
| Qwen3-4B | 4B | 0.6825 | 0.6486 |
| gte-large-en-v1.5 | 434M | 0.6322 | 0.6174 |
| Qwen3-Reranker-0.6B | 600M | 0.6162 | 0.6056 |
| deberta-v3-large-nli-triplet | 434M | 0.6131 | 0.5958 |
| e5-base-v2 | 110M | 0.6106 | 0.5966 |
| e5-large-v2 | 335M | 0.6060 | 0.5974 |
| deberta-v3-large-nli | 434M | 0.5990 | 0.5911 |
| gte-base-en-v1.5 | 137M | 0.5910 | 0.5842 |
| modernbert-large-nli | 395M | 0.5865 | 0.5517 |
| gte-reranker-modernbert-base | 149M | 0.5829 | 0.5785 |
| Qwen3-Embedding-0.6B | 600M | 0.5820 | 0.5797 |
| bge-base-en-v1.5 | 137M | 0.5791 | 0.5683 |
| deberta-v3-base-nli | 184M | 0.5778 | 0.5504 |
| gte-modernbert-base | 149M | 0.5772 | 0.5864 |
| modernbert-base-nli | 149M | 0.5752 | 0.5343 |
| modernbert-large-nli-triplet | 395M | 0.5713 | 0.5487 |
| Qwen3-Embedding-8B | 8B | 0.5654 | 0.5913 |
| bert-large-uncased-nli-triplet | 335M | 0.5600 | 0.5246 |
| e5-mistral-7b-instruct | 7B | 0.5583 | 0.5756 |
| bert-large-uncased-nli | 335M | 0.5579 | 0.5339 |
| bge-large-en-v1.5 | 434M | 0.5487 | 0.5548 |
| bge-reranker-large | 560M | 0.5378 | 0.5348 |
| bert-base-uncased-nli | 110M | 0.5245 | 0.4876 |
| bart-large-mnli | 407M | 0.5215 | 0.5079 |
| nli-roberta-base | 125M | 0.5187 | 0.4885 |
| bge-reranker-base | 278M | 0.4811 | 0.4711 |
| ms-marco-MiniLM-L6-v2 | 22M | 0.4773 | 0.4218 |
| Phi-4-mini-instruct | 3.8B | 0.4519 | 0.4309 |
| Llama-3.2-3B-Instruct | 3.2B | 0.4362 | 0.4302 |
| gemma-3-1b-it | 1B | 0.4104 | 0.3591 |
| all-MiniLM-L6-v2 | 22M | 0.3745 | 0.3660 |
| gemma-3-270m-it | 270M | 0.3352 | 0.2794 |
