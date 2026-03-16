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
