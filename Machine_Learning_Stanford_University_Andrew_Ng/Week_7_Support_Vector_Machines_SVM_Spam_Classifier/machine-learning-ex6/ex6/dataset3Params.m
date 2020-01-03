function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%########################## Renuka ##########################
% Algorithm. 
% Step 1) setup c Values and Sigma values as expected by ex6.pdf. Also initialize
% variables to track minimum error and corresponding sigma and c values
% step 2) step thru each of those two in a pair and 
% train, and predict 
% step 3) find error - we wiull mean prediction error mean(predictions - y)
% step 4) if error is less than minimum_error store current c and sigma values 

%step 1
c_values = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_choices = c_values;
minimum_error = inf;
minimum_error_at_c = inf;
minimum_error_at_sigma = inf;


for C = c_values
	for sigma = sigma_choices
    % step 2
		model = svmTrain(X, y, C, @(x1, x2) ... 
              gaussianKernel(x1, x2, sigma));
		predictions = svmPredict(model,Xval);
    %step 3
		prediction_error = mean(double(predictions ~= yval));
    % step 4
		if prediction_error < minimum_error
			minimum_error = prediction_error;
		  minimum_error_at_c = C;
			minimum_error_at_sigma = sigma;
		end
	end
end		

% we are done
C = minimum_error_at_c;
sigma = minimum_error_at_sigma;

% =========================================================================

end
