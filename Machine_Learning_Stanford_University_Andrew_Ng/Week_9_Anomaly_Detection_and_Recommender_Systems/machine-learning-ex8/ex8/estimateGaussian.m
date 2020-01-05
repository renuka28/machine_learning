function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

%################# Renuka ##########################
%we can do a simpler implmentatoin using hte built in functions using below code.
%I am skipping this to do vectorized implmentation for educational purpose
%mu = (mean(X))'
%sigma2 = var(X,1)
% Trying vecorized implementation. 
% step 1) create a ones matrix with m rows of 1
ones_mat = ones(m, 1);
% step 2) multiply the X with ones mat and divide by lenght to get mean
mu = ((X' * ones_mat) ./m);
% step 3) subtract each element of X by mu, square it and divide the sum by
% total elements to get sigma
sigma2 = (sum((X .- mu').^2)./m)';

% =============================================================


end
