function sim = gaussianKernel(x1, x2, sigma)
%RBFKERNEL returns a radial basis function kernel between x1 and x2
%   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
%   and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);

% You need to return the following variables correctly.
sim = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the similarity between x1
%               and x2 computed using a Gaussian kernel with bandwidth
%               sigma
%
%

%############### Renuka ################################
%algorithm
% step 1) find eculidean distance between x1 and x2 (landmark)
euclidean_distance = x1-x2;

% step 2) square and sum the euclidean distance
squared_sum_of_euclidean_distance = sum((x1-x2) .^2);
% step 3) dividesquared_sum_of_euclidean_distance 2*sigma*square and take its
% negative exp
sim = exp(-(squared_sum_of_euclidean_distance)/(2*sigma^2));



% =============================================================
    
end
