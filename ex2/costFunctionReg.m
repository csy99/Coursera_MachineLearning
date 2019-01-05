function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = size(theta);

% You need to return the following variables correctly 
J = 0;
J2 = 0;
grad = zeros(n);
h = zeros(m);
grad = zeros(n);
gradtmp = zeros(n);

for i = 1:m
  h_theta = 1/(1+exp(-theta' * X(i,:)')); %this is a number 
  h(i) = h_theta;
  J += y(i)*log(h_theta) + (1 - y(i)) * log(1-h_theta);
endfor
J = -J/m;

for j = 2:n
  J2 += theta(j)^2  ;
endfor
J = J + J2*lambda/(2*m);

for itr = 1:n
  for i = 1:m
    gradtmp(itr) += (h(i) - y(i))*X(i,itr);
  endfor
  gradtmp(itr) += lambda*theta(itr);
endfor
grad = gradtmp/m;

%we should not regularize theta(1)
grad(1) = grad(1) - lambda*theta(1)/m; 

end
