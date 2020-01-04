function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
% ############# REnuka ################





% step 1) loop over all X values
for i = 1:size(idx)  
  % step 2) initialize to inf
  min_norm = inf;
  for current_centroid_idx = 1:K
      % step 3) loop over all centroid and see if any of other centroid is closer
      % if so, update hte idx vector to the centroid index which is closest
      current_norm = sum((X(i,:)-centroids(current_centroid_idx,:)) .^ 2);
      if (current_norm < min_norm)
        min_norm = current_norm;
        idx(i) = current_centroid_idx;
      end
  end
end


% =============================================================

end

