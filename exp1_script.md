# Deep Learning for Time-Series Forecasting

## Slide 1: Introduction
- **Welcome** to our study of using deep learning for time-series prediction, focusing on forward versus backward data coorelations presented by Chris Guarino, Travis Virgil, and Kevin Kaplan. 
- **Today's focus**:In our first experiement we will explore several neural network models including RNNs, LSTMs, GRUs, and Transformers and attempt to identify whether or not there is a noticable difference between prediciting future data versus past data.
- **Data source**: We use historical stock price data from Apple Inc. (AAPL) as our example. 

## Slide 2: Data Preparation
- **Data retrieval**: AAPL stock data is fetched via Yahoo Finance using `pandas_datareader` and `yfinance`.
- **Preprocessing steps**:
  - Two distinct datasets are established, a forward flowing and a backward flowing. 
  - Data is split into training and testing sets to effectively evaluate our models.
  - Features are standardized using `MinMaxScaler` to enhance model performance.

## Slide 3: Building Time-Series Sequences
- **Sequence Transformation**: To prepare our data for the sequential nature of our models, we transform our dataset into sequences.
- **Purpose**: These sequences are used to predict stock prices based on previously seen data points.
- **Advantage**: This method enhances our model's ability to capture temporal patterns and dependencies.

## Slide 4: Simple RNN Model
- **Architecture**: The first model that we ran our data through is a simple RNN:
  - This model is made up of an input layer, a recurrent layer, and a fully connected layer that outputs the prediction.
- **Training**: We train this model using backpropagation through time and optimize with the Adam optimizer.
- **Loss Metric**: Loss is calculated using Mean Squared Error (MSE) to measure the prediction accuracy.

## Slide 5: LSTM and GRU Models
- **Model Explanation**:
  - LSTM and GRU architectures are designed to better capture long-term dependencies compared to standard RNNs.
  - GRUs are simpler and potentially faster to train than LSTMs with similar performance benefits.
- **Training Conditions**: Each model is trained under the same conditions for a fair comparison.

## Slide 6: Transformer Model
- **Model Introduction**: We introduce a Transformer model adapted for time-series:
  - Utilizes positional encodings and self-attention mechanisms.
  - Aimed at capturing complex patterns over long time horizons without the need for recurrent architecture.
- **Training Focus**: Training involves adjusting weights to minimize the MSE loss, aiming for low test errors indicative of good generalization.

## Slide 7:
- **Results**: The forward test loss, represented by the blue line, demonstrates a gradual decrease, showing a slower rate of convergence over the epochs compared to the backward test loss, which remains very low (orange line) throughout the epochs. This could suggest that the model adapts more readily or effectively to the backward sequence of the data when predicting outcomes. 

The slower convergence of the forward test loss might indicate that the model is more sensitive to the initial conditions or the sequence's direction in the forward mode. It could be that certain patterns or dependencies in the data are more effectively captured when the sequence is reversed.

The nature of the time series data itself might inherently favor backward learning. This could be due to the way events are correlated or the presence of leading indicators at the end of the series that are more predictive when viewed in reverse.

- **Theory**: The RNN might be learning different aspects of the data in forward and backward modes. For instance, if the data has a strong temporal structure where recent events significantly influence the future, the reversed model could effectively "see the future" first, thus learning more efficiently.


