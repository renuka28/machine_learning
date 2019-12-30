function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================
    % ####### Renuka - algorithm #########
    % 1) do prediciton based on current theta
    % 2) compute error = predictions - actual values
    % 3) calculate new values for individual thetas (we have two theta0 and theta1)
    % 4) create new theta vector 
    % 5) use the new theta to calculate the current cost
 
    prediction = X * theta; % 1)
    prediction_error = prediction - y; % 2)
    x1 = X(:,1);
    x2 = X(:,2);
    theta1 = theta(1) - (alpha / m) * sum(prediction_error .* x1); %3) theta0
    theta2 = theta(2) - (alpha / m) * sum(prediction_error .* x2); %3) theta1
    theta = [theta1;theta2]; % 4) new theta vector
    
    % Save the cost J in every iteration        
    J_history(iter) = computeCost(X, y, theta); 
    

end

end
