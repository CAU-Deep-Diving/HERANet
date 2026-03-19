# HERANET(최우수상)

## Overview

**HERANet (Hierarchical LSTM-based Representation Network)** is a **deep learning model** for **fake news detection**, developed to classify news articles as **real** or **fake**.  
It processes each article in a **hierarchical manner** by splitting the document into **chunks**, encoding them with a **Hierarchical LSTM**, and combining them into a unified **document-level vector**.  

To enhance its ability to capture important information, HERANet employs **attention pooling**, allowing the model to assign greater weight to the most informative segments of each article.  
Through this design, the model can reflect both **local contextual information** and **global document structure**.  

This repository summarizes our **motivation**, **model architecture**, **training strategy**, and **experimental results** from a **deep learning hackathon project** in which our team won **2nd place**.

## Hackathon Information

- **Host:** CAU SW Education Institute  
- **Date:** October 1, 2025  
- **Task:** Fake News Classification  
- **Constraint:** The use of **pretrained tokenizers** was not allowed

 ## Problem Definition

As online news platforms and social media continue to grow, factual reporting and fabricated content are increasingly mixed together, making it difficult to judge credibility based on intuition alone. Traditional approaches such as keyword matching or shallow classifiers often struggle to capture deeper contextual meaning, writing style, and document-level consistency.

In this project, we address the task of **binary fake news classification**, where the goal is to determine whether a given news article is **real** or **fake**. We focus on a **small-scale dataset setting** under the competition constraint that **pretrained tokenizers were not allowed**. Our objective is to investigate whether a **Hierarchical LSTM-based model with Attention Pooling** can effectively capture both **local contextual information** and **global document structure** without relying on pretrained language models.


## Datasets

To evaluate fake news classification under different text lengths, metadata availability, and dataset scales, we used three heterogeneous datasets.  
These datasets differ in document structure, average length, and label distribution, allowing us to test whether the proposed hierarchical architecture can generalize across both long-form news articles and short-form text samples.

### Dataset Summary

| Dataset | # Samples | Columns | Delimiter | Avg. Text Length | Label Distribution | Characteristics |
|---|---:|---|---|---:|---|---|
| `dataset_1.csv` | 32,470 | `title`, `text`, `label` | semicolon (`;`) | ~2,483 chars | fake 54% / real 46% | Large-scale, article-style dataset with long news bodies |
| `dataset_2.csv` | 250 | `title`, `text`, `subject`, `date`, `label` | comma (`,`) | ~2,584 chars | fake 47% / real 53% | Cleaned news dataset with metadata such as subject and date |
| `dataset_3.csv` | 250 | `text`, `label` | comma (`,`) | ~342 chars | real 71% / fake 29% | Short-form dataset with summary-like or tweet-like texts |

### Dataset Characteristics

#### 1. `dataset_1.csv`

This is the largest dataset in our project and mainly consists of full-length news articles.  
Because the documents are relatively long, it is suitable for evaluating whether the model can capture hierarchical structure, long-range dependencies, and document-level consistency.

Its near-balanced label distribution also makes it appropriate for large-scale supervised training without severe skew toward one class.

#### 2. `dataset_2.csv`

This dataset is much smaller, but it includes additional metadata such as `subject` and `date`.  
Although our main model primarily focuses on textual content, this dataset is useful for examining how the classifier behaves on more structured and curated news articles.

Since the text length is still relatively long, it provides another testbed for validating the effectiveness of the hierarchical encoding strategy in a low-resource setting.

#### 3. `dataset_3.csv`

Unlike the other two datasets, this dataset contains short texts rather than full news articles.  
Its compact, summary-like style makes the classification task different from long-document modeling, since there is less context and weaker document structure to exploit.

In addition, the label distribution is more imbalanced, with a larger portion of real samples, making this dataset more challenging from a class-balance perspective.

### Why These Datasets Matter

The three datasets were intentionally chosen to reflect different fake news detection scenarios:

- long-form article classification
- metadata-rich but small-scale news classification
- short-text classification with imbalance

This diversity allows us to assess whether HERANet is robust across varying input conditions rather than being optimized for only one specific text format.

### Relevance to HERANet

The proposed HERANet architecture is particularly well suited for long and structured documents.  
Instead of encoding the entire article as a single flat sequence, the model divides a document into chunks, processes each chunk with an LSTM encoder, and then aggregates them into a document-level representation using attention pooling.

This design is especially beneficial for `dataset_1.csv` and `dataset_2.csv`, where article bodies are long and document structure contains meaningful information.  
At the same time, evaluation on `dataset_3.csv` helps verify whether the model can still remain effective when hierarchical structure is limited.

### Practical Challenges

Working with these datasets introduces several challenges:

- Different CSV delimiters must be handled carefully during preprocessing.
- Input lengths vary significantly across datasets.
- Some datasets are small, which increases the risk of overfitting.
- Label imbalance is present, especially in the short-text dataset.
- Because pretrained tokenizers were not allowed, all tokenization and vocabulary construction had to be performed from scratch.

These characteristics motivated us to design a model that is both structurally efficient and capable of learning useful representations under constrained settings.
Input lengths vary significantly across datasets.

Some datasets are small, which increases the risk of overfitting.

Label imbalance is present, especially in the short-text dataset.

Because pretrained tokenizers were not allowed, all tokenization and vocabulary construction had to be performed from scratch.

These characteristics motivated us to design a model that is both structurally efficient and capable of learning useful representations under constrained settings.

## Modeling Approach

To address fake news classification under the constraint that pretrained tokenizers were not allowed, we designed **HERANet**, a **hierarchical BiLSTM-based architecture** with **attention pooling**.  
The main idea is to process a document at two levels:

1. **Token level**: encode each chunk of text to capture local contextual information  
2. **Chunk level**: aggregate chunk representations to capture global document structure  

This hierarchical design allows the model to handle long news articles more effectively than a flat sequence encoder.

### Model Architecture

<p align="center">
  <img src="./assets/heranet_architecture.svg" alt="HERANet Architecture" width="100%">
</p>

<p align="center">
  <em>Overview of HERANet: a hierarchical BiLSTM architecture with attention pooling for fake news classification.</em>
</p>

### Step-by-Step Pipeline

#### 1. Text Input and Token Embedding

Each news article is first tokenized using a custom tokenizer built from scratch.  
Since pretrained tokenizers were not allowed in the competition, all vocabulary construction and token indexing were performed manually.

The input tokens are then mapped into dense vector representations through a trainable **token embedding layer**.

#### 2. Chunking

Because news articles can be long, directly feeding the entire sequence into a single recurrent model is inefficient and may weaken long-range information flow.  
To address this, each document is divided into multiple **chunks** of fixed length.

This chunking strategy makes the model more scalable and enables hierarchical document modeling.

#### 3. Token-Level BiLSTM Encoder

Each chunk is independently processed by a **Bidirectional LSTM (BiLSTM)** at the token level.  
This allows the model to capture contextual dependencies in both forward and backward directions within each chunk.

As a result, each token is transformed into a contextualized hidden state.

#### 4. Token-Level Attention Pooling

Not all words contribute equally to fake news detection.  
Therefore, after obtaining token-level hidden states, we apply **attention pooling** to compute a weighted sum of the token representations.

This produces a single **chunk representation** for each chunk, while allowing the model to focus on the most informative words.

#### 5. Chunk-Level BiLSTM Encoder

The sequence of chunk representations is then passed to another **BiLSTM**, this time at the **chunk level**.  
This layer models relationships across chunks and captures higher-level discourse flow and document structure.

In other words, while the token-level BiLSTM captures local semantics, the chunk-level BiLSTM captures global context.

#### 6. Chunk-Level Attention Pooling

After chunk-level encoding, we again apply **attention pooling** over the sequence of chunk representations.  
This step allows the model to identify which parts of the document are most important for classification.

The resulting weighted representation becomes the final **document embedding**.

#### 7. Classification

The document embedding is passed through **dropout** for regularization and then fed into a **linear classifier** to predict whether the article is **fake** or **real**.

### Why Hierarchical BiLSTM?

A flat text encoder may struggle when handling long news articles because important evidence can be distributed across multiple parts of the document.  
By contrast, the hierarchical structure in HERANet offers the following advantages:

- it captures both **local token-level context** and **global document-level structure**
- it is more suitable for **long-form news articles**
- it reduces the difficulty of modeling extremely long token sequences at once
- it enables more interpretable aggregation through **attention pooling**

### Why Attention Pooling?

Attention pooling is used at both the token level and the chunk level because the importance of textual units is not uniform.  
Some words or chunks contain stronger signals for fake news detection than others.

By assigning larger weights to more informative tokens and chunks, the model can build a more discriminative document representation.

### Summary

In summary, HERANet is a **two-stage hierarchical encoder**:

- **Token-level BiLSTM + Attention Pooling** to generate chunk representations
- **Chunk-level BiLSTM + Attention Pooling** to generate a document representation
- **Dropout + Linear Classifier** for final fake/real prediction

This architecture is particularly effective for long and structured news documents, while still remaining lightweight enough to operate without pretrained language models.

