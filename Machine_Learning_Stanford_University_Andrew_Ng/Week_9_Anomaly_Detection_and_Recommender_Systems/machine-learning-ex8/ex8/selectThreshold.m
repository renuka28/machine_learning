function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;


stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    %############# Renuka ####################
    %Step 1) find predictions. any less than epsilon is anamoly another more is 
    % fine
    predictions = (pval < epsilon);  
    %step 2) calculate false_positives - which is the sum of what algorithm predicts
    % as positive but actually negative    
    fp = sum((predictions == 1) & (yval == 0));
    %step 3) calculate false negative - sum of what algorithm predicts as negatives
    % but actually positive
    fn = sum((predictions == 0) & (yval == 1));
    % step 4) ture positive - sum of what algorithm predicts as positive and is
    % indeed positive
    tp = sum((predictions == 1) & (yval == 1));
    % step 5) formula for precision from ex8.pdf
    prec = tp / (tp + fp);
    % step 5) formula for recall from ex8.pdf
    rec = tp / (tp + fn);
    % step 7) formula for F1 score from ex8.pdf
    F1 = (2*prec*rec) / (prec + rec);
  
    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
