function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);
K = num_labels;

% You need to return the following variables correctly 
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
a2 = zeros(m, size(Theta1,1));
Z2 = zeros(m, size(Theta1,1));
X = [ones(m, 1) X];
a3 = zeros(m, K);
Z3 = zeros(m, K);

Z2 = X*Theta1';
a2 = sigmoid(Z2);

a2 = [ones(size(a2,1), 1) a2]; %add bias unit to hidden layer
Z3 = a2*Theta2';

a3 = sigmoid(Z3);

[value, idx] = max(a3,[],2);
p = idx;



% =========================================================================


end