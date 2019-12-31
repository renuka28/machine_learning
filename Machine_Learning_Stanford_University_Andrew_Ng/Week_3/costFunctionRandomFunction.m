function [jVal, gradient] = costFunctionRandomFunction(theta)
  % j(theta) = (theta1 - 5)^2 + (theta2 - 5) ^ 2
  % partial derivatie of theta wrt theta1 = 2(theta1 - 5)
  % partial derivatie of theta wrt theta2 = 2(theta2 - 5)
  
  jVal = (theta(1) - 5) ^2 + (theta(2) - 5) ^ 2;
  gradient = zeros(2,1);
  gradient(1)= 2*(theta(1) - 5);
  gradient(2)= 2*(theta(2) - 5);
  
end



