options = optimset('GradObj', 'on', 'MaxIter', '10000');
initialTheta = zeros(2,1);
optTheta = fminunc(@costFunctionRandomFunction, initialTheta, options)