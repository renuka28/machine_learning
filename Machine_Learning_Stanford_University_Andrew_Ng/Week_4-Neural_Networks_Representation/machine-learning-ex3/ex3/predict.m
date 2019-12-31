function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%############# Renuka ####################
% algo
% this is a preconfigured NN with three layers. One input, one hidden and a
% final output layer. theta1 represents the weights for input layer and theta2
% weights for hidden layer. Input layer activiation (a1) is X + 1's to account 
% for bias we need to calculate hidden layer activiations (a2) which will be 
% sigmoid(a1 * theta1'). We will multiply hidden layer activiations with 
% hidden layer wieghts theta2 to get output layer activations

% in a more general implementaiton, we should have a loop going across
% all layers, activating layer by layer. since we only have two we can do it
% step by step for better understanding

% step 1) calcualte a1 - input lyaer activiations which is basically our input
% matrix with ones added to account for bias
% step 2) calcualte a2 - which is sigmoid of(a1 * theta1) - this is hidden layer
% activiation
% step 3) calcualte a3 - which is sigmoid of (a2 * theta2) - activiations of 
% final and output layer. We take the max of this to get the actual indexes
% note the output expected out of this function is the index value of the 
% maximum probablity 

% add ones to account for bias
% step 1
a1_input_layer_activiations = [ones(m, 1) X]; 
% step 2
a2_hidden_layer_activiations = sigmoid(a1_input_layer_activiations * Theta1'); 
% add ones again to account for bias
a2_hidden_layer_activiations = [ones(m, 1) a2_hidden_layer_activiations];
% step 3
a3_output_layer_activiations = sigmoid(a2_hidden_layer_activiations * Theta2' );
%choose the maximum value for reach row just like we did in multiclass logistics
% one vs All
[max_probability, max_index]  = max(a3_output_layer_activiations, [], 2);
p = max_index


% =========================================================================


end
