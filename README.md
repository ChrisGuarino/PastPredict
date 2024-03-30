# Predicting the Past: A Time Series Project

This project explores time series analysis with an emphasis on experimentation with recurrent layers and transformers. The training set is denoted as (X, Y), where each training datum Xi is a sequence of vectors:


and Yi is a label derived as a function of Xi and Xi,t+1, which can be either categorical or numerical.

## Dataset Example

Consider the dataset as weather data, where each vector Xi,j represents a set of measurements from a weather station over time, and the ‘datum’ Xi is a sequence of these measurements. Yi may represent a vector of measurements at t + 1 or a general categorization of the weather for time t + 1.

## Experimentation

We will explore models for both forward prediction (predicting Yi = Xi,t+1) and augment our target with the following experiments:

### Experiment-1: Predicting the Past

- **Objective**: Reverse the sequence to {Xi,t, ..., Xi,1}, with the label derived from Xi,0 (e.g., Yi = Xi,0), to explore the concept of 'predicting the past.'
- **Questions**:
  - How does the model perform with this setup?
  - Is there an improvement in performance or faster convergence?
  - Philosophically, does this experiment suggest an arrow of time?

### Experiment-2: Dual Time Direction Prediction

- **Objective**: Given a sequence {Xi,1, ..., Xi,t}, predict both Xi,1 and Xi,t from the rest of the sequence. Model the total loss L as the sum of two losses, L1 and L2.
- **Questions**:
  - How does the loss behavior reflect the arrow of time?
  - Does L1 reduce faster than L2, or do they reduce at the same rate?

### Experiment-3: Practical Relevance of Predicting the Past

- **Objective**: Train a model to minimize L, predicting both the past and the future, and analyze the correlation between backward and forward prediction accuracy during test/evaluation.
- **Questions**:
  - Is there a correlation between the accuracy of backward and forward predictions?
  - Can the accuracy of past predictions during deployment provide a degree of certainty for future predictions or indicate market volatility?

## Models

Experiments will be conducted using relatively simple networks, incorporating
