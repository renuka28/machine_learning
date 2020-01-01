function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);

% ====================== YOUR CODE HERE ======================
% Instructions: Initialize W randomly so that we break the symmetry while
%               training the neural network.
%
% Note: The first column of W corresponds to the parameters for the bias unit
%


% ##### Renuka ############ 
% code is given in the ex4.pdf. just enter it here. instead of hardcoding epsilon
% i have used the formula given in the footnote on page 7 of ex4.pdf to calculate
% the same. Formula is ep_init = sqrt(6)/sqrt(input_layer_nueron_count + next_layer_neuron_count)
% this works out to be sqrt(6)/sqrt(400 + 25). we ahve 400 (20*20) neurons in 
% the input layer and 25 in the next one

epsilon_init = sqrt(6)/sqrt(400 + 25)

W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;


% =========================================================================

end
