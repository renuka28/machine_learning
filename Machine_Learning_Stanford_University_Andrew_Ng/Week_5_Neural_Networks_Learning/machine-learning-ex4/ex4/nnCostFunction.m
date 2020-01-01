function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%############# Renuka ####################
% Part 1 algorithm - Calculate J
% this is a preconfigured NN with three layers. One input, one hidden and a
% final output layer. theta1 represents the weights for input layer and theta2
% weights for hidden layer. Input layer activiation (a1) is X + 1's to account 
% for bias we need to calculate hidden layer activiations (a2) which will be 
% sigmoid(a1 * theta1'). We will multiply hidden layer activiations with 
% hidden layer wieghts theta2 to get output layer activations

% in a more general implementaiton, we should have a loop going across
% all layers, activating layer by layer. since we only have two we can do it
% step by step for better understanding
% 
% step 1) calcualte a1 - input lyaer activiations which is basically our input
% matrix with ones added to account for bias
% step 2) calcualte a2 - which is sigmoid of(a1 * theta1) - this is hidden layer
% activiation
% step 3) calcualte a3 - which is sigmoid of (a2 * theta2) - activiations of 
% final and output layer. We take the max of this to get the actual indexes
% note the output expected out of this function is the index value of the 
% maximum probablity 
% step 4) map given y lable vector into a a binary vectory. We can use
% octave's repmat function to do this
% step 5) calculate J
% step 1 to 3 is a copy and paste from previous week implementation

% -------------------------------------------------------------

% add ones to account for bias
% step 1
a1_input_layer_activiations = [ones(m, 1) X]; 
% step 2
a2_hidden_layer_activiations = sigmoid(a1_input_layer_activiations * Theta1'); 
% add ones again to account for bias
a2_hidden_layer_activiations = [ones(m, 1) a2_hidden_layer_activiations];
% step 3
a3_output_layer_activiations = sigmoid(a2_hidden_layer_activiations * Theta2' );
% step 4 - we basically have to create a truth table where the vector (of size k)
%has value 1 where the lable is equal to corresponding k value and 0 every where
% else. 
%create a matrix of which has m rows of all lables. 
label_mat = repmat([1:num_labels], m, 1);
%this will create a matrix of y's (1 * num_lables)
y_mat = repmat(y, 1, num_labels);
%truth table 
y = label_mat == y_mat;
% step 5) we have everything to calculate J now
J = (-1 / m) * sum(sum(y .* log(a3_output_layer_activiations) + ...
                      (1 - y) .* log(1 - a3_output_layer_activiations)));

% updating to add regularization. 
% we don't apply regularization for theta0. So we will copy to a separate
%vector all thetas starting from second one.            
theta1_for_regularization =  Theta1(:,2:end);
theta2_for_regularization =  Theta2(:,2:end);

regularization_factor = (lambda/(2*m)) * ...
      (sum(sum(theta1_for_regularization .^ 2)) + ... 
        sum(sum(theta2_for_regularization .^ 2)));

J = J + regularization_factor;

%choose the maximum value for reach row just like we did in multiclass logistics
% one vs All
%[max_probability, max_index]  = max(a3_output_layer_activiations, [], 2);
%p = max_index

% implement backpropogation algo

DELTA1 = zeros(size(Theta1));
DELTA2 = zeros(size(Theta2));
for i = 1:m,
  % i have  modified a bit the algo given in the ex4.pdf. I have not calculated
  % the intermediary terms. went by my intution, which is calculating from 
  % backwards. So step 1 from ex4 is mssing
  % calculate the error at the output. differnece between activations at output
  % layer - actual y values - Step 2 in ex4
  delta_3_output_error = a3_output_layer_activiations(i, :) - y(i,:);
  % calculate error at the hidden layer - Step 3 in ex4
  delta_2_hidden_layor_error = Theta2' *delta_3_output_error' ... 
              .* sigmoidGradient([1;Theta1 * a1_input_layer_activiations(i,:)']);
  % update our deltas for current training example - step 4 from ex4
  DELTA1 = DELTA1 + delta_2_hidden_layor_error(2:end)*a1_input_layer_activiations(i,:);
	DELTA2 = DELTA2 + delta_3_output_error' * a2_hidden_layer_activiations(i,:);
  
end;

%Finally we are ready to calculate our gradients. Man that was tough and 
% hours of work - step 5 from ex4
Theta1_grad = 1/m * DELTA1 + (lambda/m)*[zeros(size(Theta1, 1), 1) theta1_for_regularization];
Theta2_grad = 1/m * DELTA2 + (lambda/m)*[zeros(size(Theta2, 1), 1) theta2_for_regularization];

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
