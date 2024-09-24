

# Amazon Reviews Sentiment Classification


<p align="center">
<img src="AmazonRev.png"
     alt="Markdown Monster icon"
     style=" padding:30px ;  width:400px" />
</p>
    The classification model implemented by  a Recurrent Neural Network (RNN) with LSTM (Long Short-Term Memory) layers which classifies the  Amazon product reviews as either positive (1) or negative (0)

## Overview

- Sentiment analysis is a common task in Natural Language Processing (NLP), where the goal is to classify a piece of text based on the sentiment it conveys.The model is trained on a dataset of Amazon product reviews where each review is labeled as either positive or negative.
## Dataset

- *source*: [AmazonReviews](https://www.kaggle.com/datasets/mahmudulhaqueshawon/amazon-product-reviews/data)
- *Dataset dimensions*: 19k entries x 2 columns
- *Problem Type*: Classification




##### Embedding Layer:

- This layer converts the input text (encoded as integers) into dense vector representations of fixed size (embedding_dim). The model FakeNewsRNN is built using the following components:

##### LSTM Layers:

- The core of the RNN architecture, LSTM layers are used to capture long-range dependencies and relationships in the text data. The LSTM processes the input sequence, and the hidden states from the last time step are passed to the fully connected layers for classification.

##### Fully Connected Layers:

- After the LSTM layers, the model includes two fully connected layers that gradually reduce the dimensionality of the data. Each layer is followed by batch normalization and dropout to prevent overfitting and speed up convergence.

##### Output Layer:
- The final layer outputs a single value (positive or negative) after passing through the fully connected layers.

##### Regularization:
- Dropout is applied to prevent overfitting, and L2 regularization is used in the optimizer.


#### Key Parameters:
* no_layers: Number of LSTM layers (2 in this case).
* vocab_size: The size of the vocabulary derived from the dataset.
* hidden_dim: Number of hidden units in the LSTM layer (512).
* embedding_dim: The size of the word embeddings (128).
* output_dim: The size of the output, which is 1 for binary classification (positive/negative).




<img src="LSTM.png"
     alt="Markdown Monster icon"
     style="  float: left; margin-right: 10px; width:500px" />

##### Forward Pass:

* The input review is first converted into embeddings using the nn.Embedding layer.
* The embeddings are passed through the LSTM layers, where the hidden states capture the context of the review.
* The final hidden state is passed through fully connected layers, with ReLU activations, batch normalization, and dropout applied at each stage.
* The output is a single value that is passed through a sigmoid function to determine the probability of the review being positive.

#### Training
##### Loss Function:
- Binary Cross Entropy with Logits (BCEWithLogitsLoss): This is used for binary classification tasks where the output is a single value between 0 and 1.
##### Optimizer:
- Adam optimizer with L2 regularization (weight_decay = 0.01) to prevent overfitting and ensure smooth convergence.
    Learning rate is set to 0.0001.

<p align="center">
<img src="image.png"
     alt="Markdown Monster icon"
     style="margin_left:50px; margin-right: 10px; width:500px;" />

</p>

##### How to Run
* Clone the repository and install the required dependencies.
* Prepare the dataset of Amazon reviews and preprocess the text data (e.g., tokenization, padding).
* Initialize the model using the makeTheNet() function, which returns the model, loss function, and optimizer.
* Train the model using your dataset, and evaluate its performance on a validation or test set.


## Model Performance Evaluation

<p align="center">
<img src="ModelPer.png"
     alt="Markdown Monster icon"
     style="margin_left:50px; margin-right: 10px; width:800px;" />

</p>

##### 1. Model Accuracy Over Epochs

* The left plot depicts the Training and Testing Accuracy curves over the course of 100 epochs.

- Observations:
* The Training Accuracy (light pink line) starts at ~60% and improves rapidly over the first 20 epochs.
* The model achieves a high training accuracy of around 88-89% by epoch 40, where the curve begins to plateau.
* The Testing Accuracy (darker pink line) follows a similar upward trend, reaching a final test accuracy of 88.28%.
* Although there are fluctuations in the testing accuracy, especially after 40 epochs, the overall trend is strong with no significant overfitting. The testing accuracy remains close to the training accuracy, indicating good model generalization.
- Insights:
* The model demonstrates solid performance in terms of accuracy, converging quickly in the early epochs and maintaining high accuracy.
* The small difference between training and testing accuracy suggests the model generalizes well to unseen data.
##### 2. Model Loss Over Epochs

* The right plot illustrates the Training and Testing Loss curves over the course of 100 epochs.

- Observations:
* The Training Loss (light pink line) starts high (around 0.65) and decreases sharply during the first 20 epochs.
* Similarly, the Testing Loss (darker pink line) follows the same trend, showing a significant reduction in the initial epochs.
* By the end of training, both losses stabilize around 0.29, although there are minor fluctuations and spikes, especially in the testing loss.

- Insights:
* The rapid reduction in loss indicates the model is learning effectively, particularly in the early epochs.
* The final loss value of 0.29 is a good indicator of model performance, though the minor fluctuations in the testing loss suggest there could still be room for fine-tuning.
* The spikes in the testing loss may be due to inherent variability in the dataset or a result of overfitting to certain aspects of the training data. Further regularization or learning rate adjustments could help smooth the loss curve.
##### 3. Final Model Evaluation
- Final Testing Accuracy: 88.28%
- Final Testing Loss: 0.29
* The high accuracy and relatively low loss indicate that the model is effective in classifying reviews with high confidence. While there are fluctuations in both loss and accuracy, the overall trend shows strong learning capability and generalization to the testing set.

##### 4. Potential Areas for Improvement
* While the model performs well, the following areas could be explored for further optimization:

* Regularization: Increasing dropout rates or applying L2 regularization more aggressively could reduce fluctuations in loss.
* Learning Rate: Fine-tuning the learning rate could improve stability during training and smooth out the loss curve.
* Early Stopping: Implementing early stopping based on validation performance may prevent unnecessary fluctuations and prevent overfitting.
