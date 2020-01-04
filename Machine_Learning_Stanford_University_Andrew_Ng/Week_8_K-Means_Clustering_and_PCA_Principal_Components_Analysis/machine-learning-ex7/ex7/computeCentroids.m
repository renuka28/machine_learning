function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);


% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

%######## Renuka ##################
%step 1) run through all the k's we have
for i = 1:K  
  % step 2) find the bool vector which has identifies indexes 
  % where idx == current k (current centroid)
  vals_in_current_centroid_idx = (idx == i);
  % step 3) find x values using hte above index and sum them
  vals_in_current_centroid = X(vals_in_current_centroid_idx,:);
  % step 4) find the total number of elements with current centroid
  total_elements_in_current_centroid = sum(vals_in_current_centroid_idx);
  % step 5) calculate mean value
  centroids(i, 1:n) = sum(vals_in_current_centroid) ./ total_elements_in_current_centroid;
end

% =============================================================

end

