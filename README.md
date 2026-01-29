# PastPredict

Exploring time series prediction with recurrent and attention-based models, with experiments on reversed (past-predicting) sequences to investigate directional asymmetry in temporal learning.

## Overview

This project trains sequence models (LSTMs, GRUs, self-attention layers) on time series data and investigates whether predicting backwards in time differs fundamentally from predicting forwards. The core question: does reversing the target direction improve convergence, and can backward prediction accuracy serve as a confidence signal for forward predictions?

## Experiments

### Experiment 1: Predicting the Past
Reverse the input sequence and predict the earliest observation. Compare performance and convergence speed against standard forward prediction.

### Experiment 2: Dual Time Direction
Predict both the first and last elements of a sequence simultaneously. Analyze whether the two loss components (L1 for past, L2 for future) decrease at different rates, revealing temporal asymmetry.

### Experiment 3: Practical Relevance
Train a joint forward-backward model and measure the correlation between backward and forward prediction accuracy at test time. Investigate whether past-prediction confidence can indicate reliability of future predictions.

## Models

- LSTM
- GRU
- Self-attention layers

## Project Structure

```
PastPredict/
├── chris/           # Experiments by Chris
├── Kevin/           # Experiments by Kevin
├── travis/          # Experiments by Travis
├── exp1_script.md   # Experiment 1 methodology notes
└── README.md
```

## Resources

- [Working Report](https://drive.google.com/drive/folders/10IOqhsqau_EjCnfpGv765CGTIovB9zo5?usp=drive_link)
