# HERANET(최우수상)

## Overview

**HERANet (Hierarchical LSTM-based Representation Network)** is a **deep learning model** for **fake news detection**, developed to classify news articles as **real** or **fake**.  
It processes each article in a **hierarchical manner** by splitting the document into **chunks**, encoding them with a **Hierarchical LSTM**, and combining them into a unified **document-level vector**.  

To enhance its ability to capture important information, HERANet employs **attention pooling**, allowing the model to assign greater weight to the most informative segments of each article.  
Through this design, the model can reflect both **local contextual information** and **global document structure**.  

This repository summarizes our **motivation**, **model architecture**, **training strategy**, and **experimental results** from a **deep learning hackathon project** in which our team won **2nd place**.
