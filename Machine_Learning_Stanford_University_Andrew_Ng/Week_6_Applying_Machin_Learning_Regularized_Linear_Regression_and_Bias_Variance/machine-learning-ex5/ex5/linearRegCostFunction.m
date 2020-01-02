function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%###################### Renuka #############################
% Algorithm is similar to one without regularization with one change
% add a factor to account for regulairzation for all except theta 0
% - Calculate prediction 
% - substitue the values in the cost function forumla
% this is the same cost function we wrote in logistics regression 
%(costFunctionReg.m) under ex2 folder. Just copy pasted the same.. 

% we don't apply regularizatoin for theta0 (theta(10 in ocatave)
regTheta = theta;
regTheta(1) = 0;

%calcualte regularization factor - just substitue in formula
regularization_factor = (lambda / (2 * m))*sum(regTheta.^2);

% find prediction. no sigmoid function here as it is lineare regression
predictions = X * theta;
%prediction error is straightforward. 
prediction_error = predictions - y;

% substitute all the knownn values in the cost function 
J = (1 / (2*m)) * sum(prediction_error .^2) + regularization_factor;

%caclualtes of gradient is also striaght foward now. 
grad = (1 / m) * (X' * prediction_error) + (lambda/m)*regTheta; 


% =========================================================================

grad = grad(:);

end
