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
C_range = [0.01 0.03 0.1 0.3 1 3 10 30];
sigma_range = [0.01 0.03 0.1 0.3 1 3 10 30];
predictError = zeros(size(C_range,2), size(sigma_range,2)); % use a matrix to record prediction errors

for itr1 = 1:size(C_range,2)
  for itr2 = 1:size(sigma_range,2)
    model = svmTrain(X,y,C_range(itr1),@(x1,x2) gaussianKernel(x1,x2,sigma_range(itr2)));
    prediction = svmPredict(model,Xval);
    predictError(itr1,itr2) = mean(double(prediction ~= yval));
  endfor
endfor

% find the index of the minimum prediction error
[m,n] = find(predictError == min(min(predictError))); 
C = C_range(m);
sigma = sigma_range(n);

% =========================================================================

end
