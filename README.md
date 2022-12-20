# Semi-NAT
The idea of this project is to let the decoder predict the target sentence segment autoregressively, 
instead of predicting a single token from left to right to achieve a balance between decoding speed and accuracy.


Semi-autoregressive uses auto-regressive prediction of target sentence segments to achieve target tokens dependency modeling, 
but if too many target sentence segments are divided, it will approach the token by token autoregressive decoding method, 
and the decoding speed cannot be improved; if too few target sentence segments are divided, the target tokens dependency cannot be improved. 
for effective modeling, therefore, an appropriate segment should be selected. 
In addition, there are two ways to select segments: continuous target tokens and random target tokens. 
The impact of different selection methods on the results needs to be verified experimentally.


Specifically, the token index of a sentence is randomly shuffled and divided into n parts. 
To predict the first segment, mask the entire sentence and feed into the decoder, that is to say, the first segment prediction does not see any tokens, 
and calculate the loss of the model predicting the first segment. For the second prediction, 
the input sentence at the decoding side can see the tokens of the first segment, the tokens of other segments are masked, 
and the model predicts the loss of the second segment, and so on.


# Problem
1 In the case that the performance of NAT is already close to that of AT, the innovation and application value of semi-autoregressive are almost meaningless.


2 At present, there have been articles that have done semi-autoregression. Although the implementation method is not exactly the same, it further loses the innovation of the idea.


# Usage
How to preprocess corpus, train model and inference, please refer to NATbase repository.
