# DeepLearning_Hackton
This repository presents our fake news classification project, developed for a deep learning hackathon where our team won 2nd place. We used a Hierarchical LSTM with Attention Pooling to classify news articles as true or false, and organized our model design, experiments, and performance results in this repository.

## Overview

HERANet (Hierarchical LSTM-based Representation Network) is a fake news detection model designed to classify news articles as real or fake. The model splits each article into chunks, encodes them through a hierarchical LSTM structure, and aggregates the resulting representations into a document-level vector.  To better capture the most informative parts of each article, HERANet incorporates an attention pooling mechanism that assigns higher weights to important segments. This repository summarizes our motivation, model design, training strategy, and experimental results from a deep learning hackathon project in which our team achieved 2nd place.
