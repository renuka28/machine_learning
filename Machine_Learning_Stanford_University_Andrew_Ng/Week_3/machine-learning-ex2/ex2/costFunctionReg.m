function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% Algorithm is similar to one without regularization with one change
% add a factor to account for regulairzation for all except theta 0
% - Calculate prediction using sigmoid function
% - substitue the values in the cost function forumla

regTheta = theta;
regTheta(1) = 0

regularization_factor = (lambda / (2 * m))*sum(regTheta.^2)

predictions = sigmoid(X * theta);
J = (-1 / m) * sum(y.*log(predictions) + (1 - y).*log(1 - predictions)) + regularization_factor;
prediction_error = predictions - y;
grad = (1 / m) * (X' * prediction_error) + (lambda/m)*regTheta; 



% =============================================================

end
