function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
K = num_labels;
         
% You need to return the following variables correctly 
J = 0;
J_Reg = 0; % regularization of parameters
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
y_tmp = zeros(size(Theta2), m); % we need to recode each y value to a vector of 10*1

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Recode y
for i = 1:m
  y_tmp(y(i), i) = 1;
endfor

% FP
X = [ones(m, 1) X];
a1 = zeros(m, input_layer_size); % this is "a" in the input layer
a2 = zeros(m, size(Theta1,1)); % this is "a" in the second layer
Z2 = zeros(m, size(Theta1,1));
a3 = zeros(m, K); % this is "a" in the third layer
Z3 = zeros(m, K); 

a1 = [ones(size(a1,1), 1) a1]; %add bias unit to input layer
Z2 = X*Theta1';
a2 = sigmoid(Z2);
a2 = [ones(size(a2,1), 1) a2]; %add bias unit to hidden layer
Z3 = a2*Theta2';
a3 = sigmoid(Z3);

% cost function 
for i = 1:m
  J += -1/m * ( log(a3(i,:))*y_tmp(:,i) + log(1-a3(i,:))*(1-y_tmp(:,i)) );
endfor

% Regularization. Assume there are only two layers (excluding the input layer). 
for itr1 = 1:size(Theta1,1)
  for itr2 = 2:size(Theta1,2)
      J_Reg += (Theta1(itr1,itr2))^2;
  endfor
endfor

for itr1 = 1:size(Theta2,1)
  for itr2 = 2:size(Theta2,2)
      J_Reg += (Theta2(itr1,itr2))^2;
  endfor
endfor

%TODO the following code serves as the same function as the code above, but 
% does not work. Why?
##J_Reg += sum(sum(Theta1.^2)) + sum(sum(Theta2.^2));
##tmp = (size(Theta1,1)+size(Theta2,1)); 
##J_Reg -= tmp; % should not regularize the bias unit

J += J_Reg*lambda/(2*m);


% Backpropagation
Delta_1 = zeros(size(Theta1,1),size(a1,2));
Delta_2 = zeros(size(a3,2), size(a2,2));
Z2 = [ones(size(Z2,1), 1) Z2];

  a1 = X;
  
  % output layer
  delta3 = a3' - y_tmp;
  
  % hidden layer l=2
  delta2 = (delta3' * Theta2) .* sigmoidGradient(Z2);
  
  %remove the first element of delta2
  delta2 = delta2(:,2:end);
  
  Delta_2 += delta3 * (a2);
  Delta_1 += delta2' * (a1);
  
D_2 = Delta_2./m;
D_1 = Delta_1./m;

% regularize gradients
for itr1 = 1:size(D_1,1)
  for itr2 = 1:size(D_1,2)
    if(itr2 != 1)
      D_1(itr1,itr2) += lambda/m*Theta1(itr1,itr2);
     endif 
  endfor
endfor

for itr1 = 1:size(D_2,1)
  for itr2 = 1:size(D_2,2)
    if(itr2 != 1)
      D_2(itr1,itr2) += lambda/m*Theta2(itr1,itr2);
     endif 
  endfor
endfor

Theta1_grad = D_1;
Theta2_grad = D_2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
