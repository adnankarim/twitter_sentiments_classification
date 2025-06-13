# How Far Can Classical Models Go? Sentiment Classification vs. Zero-Shot Gemma-1B

**Classical Models, FlagEmbedding, and Zero-Shot Evaluation**

**Date:** 13. June 2025

---

## Abstract

We revisit sentence-level sentiment classification on a Twitter-style corpus of 21,000 sentences labeled as *negative*, *neutral*, or *positive*. While Transformer models dominate recent benchmarks, they remain impractical for many real-world applications due to high computational costs. We therefore benchmark five non-Transformer models under a unified training regime, standardizing tokenization, normalization, and early stopping.

These include a sparse TF–IDF + logistic regression baseline, a GloVe-initialized Bi-LSTM, the Kim (2014) Text-CNN, a minimal Sentence-CNN using 1024-dimensional FlagEmbeddings, and a hybrid FlagEmbedding + Bi-LSTM model. We also evaluate zero-shot performance of the instruction-tuned Gemma-1B LLM.

The FlagEmbedding + Bi-LSTM achieves the best macro-F₁ (0.743), outperforming TF–IDF (0.707) and GloVe Bi-LSTM (0.688), showing that stronger input representations alone can recover 70% of the typical Transformer performance gap, while training in under one minute on a single GPU. In contrast, Gemma-1B performs poorly in zero-shot (macro-F₁ = 0.65), particularly on the *neutral* class.

---

## Introduction

Sentiment analysis remains a canonical benchmark for evaluating sentence-level representations. While large Transformer models achieve state-of-the-art results, their computational demands, memory requirements, and privacy concerns limit their viability in edge devices and regulated environments. This work investigates how far **classical neural architectures** can be pushed when equipped with:

1. Robust regularization
2. Disciplined early stopping
3. Modern sentence embeddings that capture semantic structure beyond surface-level statistics

We also assess the **zero-shot** performance of Gemma-1B, an instruction-tuned language model, to evaluate the effectiveness of generic prompting for sentiment classification.

## Related Work

Early neural approaches to sentiment classification employed convolutional or recurrent networks over static word embeddings such as GloVe. While effective at the time, these models offered limited semantic depth. Subsequent research introduced sentence embeddings trained with contrastive objectives—such as SBERT and BGE—which provide stronger semantic priors and improved generalization, often without requiring task-specific fine-tuning.

FlagEmbedding, derived from the Bilingual Generalized Encoder, produces 1024-dimensional sentence representations that perform competitively with Transformer-based models while significantly reducing inference cost. We incorporate these embeddings into shallow neural architectures, extending prior work that has largely focused on token-level feature upgrades.

Zero-shot evaluation of instruction-tuned large language models, including Gemma-1B, has demonstrated potential across various NLP tasks. However, in domain-specific settings like sentiment classification, such models often require careful prompt engineering and calibration to reach acceptable performance.

---

## Dataset and Preprocessing

### Dataset

The dataset consists of 14,705 training, 3,152 validation, and 3,151 test examples, each representing a single tweet labeled as *negative*, *neutral*, or *positive*. Class distributions are shown below:

| Split | Negative | Neutral | Positive |
| :---: | :------: | :-----: | :------: |
| Train |   4,901  |  4,898  |   4,906  |
| Valid |   1,050  |  1,051  |   1,051  |
|  Test |   1,050  |  1,051  |   1,050  |

### Pre-Processing

* Unicode NFKD normalization + lower-casing
* Anonymize mentions/URLs: `<USER>`, `<URL>`
* Emojis → textual descriptors
* Compress repeated characters (≥3 → 2)
* Remove punctuation except apostrophes
* De-hash hashtags
* Collapse whitespace
* Negation propagation (`NOT_` prefix until punctuation)
* Vocabulary: top 20,000 tokens; unseen → `<OOV>`

---

## Models and Training

### Model Descriptions

We implement five non-Transformer architectures plus a zero-shot Gemma-1B baseline:

1. **TF–IDF + Logistic Regression**

   * (1,2)-gram TF–IDF, sub-linear scaling, min DF=2
   * Grid search over \$C\in{0.25,0.5,1,2}\$, L1/L2

2. **Bi-LSTM + GloVe**

   * 100-dim GloVe (6B), 2-layer Bi-LSTM (64 units/direction)
   * Global max-pooling, dropout=0.5

3. **Kim Text-CNN**

   * GloVe input
   * Conv filters widths 3,4,5×100 maps + ReLU, max-pooling, dropout=0.5

4. **Sentence-CNN**

   * Frozen 1024-dim FlagEmbedding input
   * 1D conv (kernel=1 →512 dims), adaptive max-pooling, dropout=0.5

5. **FlagEmbedding + Bi-LSTM**

   * Project 1024→192 dims, Bi-LSTM (96 units/direction), ReLU, dropout=0.5

6. **Gemma-1B Zero-Shot**

   * Prompt: system message + tweet; decode single token → class by prefix

### Training Details

* Optimizer: Adam, lr=1e-3, weight decay=1e-5
* Batch size: 128 (word models), 8 (Gemma-1B inference)
* Max epochs=30; early stopping on val macro-F₁ (patience=20, Δ=1e-4)
* Word models: max len=40 tokens
* Sentence models: full 1024-dim vectors

---

## Results and Analysis

| Model                          |  Test F₁  |
| ------------------------------ | :-------: |
| BERT-base-uncased              | **0.802** |
| Transformer (bge-base-en-v1.5) |   0.797   |
| DistilBERT-multilingual        |   0.759   |
| FlagEmbedding + Bi-LSTM        |   0.743   |
| TF–IDF + LogReg                |   0.707   |
| GloVe + Bi-LSTM                |   0.688   |
| Kim Text-CNN                   |   0.675   |
| Gemma-1B Zero-Shot             |   0.650   |
| Sentence-CNN (1×1 BGE)         |   0.424   |

Transformer models remain state of the art. Among non-Transformers, FlagEmbedding + Bi-LSTM leads (0.743), showing rich embeddings can recover large performance with minimal training. TF–IDF remains competitive (0.707). Gemma-1B zero-shot underperforms, especially on *neutral* tweets.

### Detailed Gemma-1B Classification Report

* Negative: recall 87%
* Neutral: recall 35%, F₁=0.47
* Positive: precision & recall \~70–79%, F₁=0.74
* Macro-F₁=0.65, accuracy=67%
* 382 neutral tweets misclassified as negative, indicating bias from prompt-based inference on noisy text.

---

## Architecture-Specific Failure Modes

* **GloVe Bi-LSTM**: misclassifies sarcasm/implicit positivity as neutral
* **Kim CNN**: sensitive to informal spellings (elongations)
* **Sentence-CNN**: defaults to neutral class
* **FlagEmbedding + Bi-LSTM**: reduces negative→neutral confusion by 4.6 points vs. GloVe

---

## Discussion

Upgrading from 100-dim GloVe to 1024-dim FlagEmbedding yields +5.8 pts on val macro-F₁ with fewer parameters (1.2M vs. 3.5M). Classical CNNs overfit early (peak at epoch 5). Gemma-1B zero-shot (0.650) struggles without adaptation, defaulting to negative in ambiguous cases.

---

## Limitations and Future Work

* Excludes fine-tuned Transformer baselines to focus on lightweight architectures.
* Future: incorporate learned negation detectors, advanced prompts or few-shot for zero-shot models.

---

## Conclusion

Our best classical model (FlagEmbedding + Bi-LSTM) attains macro-F₁=0.743 under one minute of training and 1.2M parameters, recovering \~70% of the Transformer gap. Gemma-1B zero-shot only achieves 0.650, indicating instruction tuning alone is insufficient for nuanced, noisy sentiment tasks.

---

## References

Full implementation and scripts: [GitHub Repository](https://github.com/adnankarim/twitter_sentiments_classification)


