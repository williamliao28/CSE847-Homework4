function [weights] = logistic_train(data, labels, mode, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
%   data    = n * (d+1) matrix with n samples and d features, where
%             column d+1 is all ones (corresponding to the intercept term)
%   labels  = n * 1 vector of class labels (taking values 0 or 1)
%   epsilon = optional argument specifying the convergence
%             criterion - if the change in the absolute difference in
%             predictions, from one iteration to the next, averaged across
%             input features, is less than epsilon, then halt
%             (if unspecified, use a default value of 1e-5)
%   maxiter = optional argument that specifies the maximum number of
%             iterations to execute (useful when debugging in case your
%             code is not converging correctly!)
%             (if unspecified can be set to 1000)
%   mode    = itereration scheme for approximating the weights. Possible
%             options are 
%             'GD':   gradienet descent (default)
%             'IRLS': Newton-Raphson
%
% OUTPUT:
%   weights = (d+1) * 1 vector of weights where the weights correspond to
%             the columns of "data"
%

if nargin < 2
    error("Require at least 2 inputs.")
end

switch nargin
    case 2
        mode    = "GD";
        epsilon = 1e-5;
        maxiter = 1000;
    case 3
        epsilon = 1e-5;
        maxiter = 1000;
    case 4
        maxiter = 1000;
end

[X_row_size, X_col_size] = size(data);
weights = zeros(X_col_size, 1);

if mode == "GD"
    eta = 3; % learning rate/ step size
    ii = 1;
    tol = inf;
    while ii <= maxiter && tol > epsilon
        w_prev  = weights;
        weights = w_prev - (eta*data.'*(sigmoid(data*weights)-labels))/X_row_size;
        ii = ii + 1;
        tol = sum(abs(data*(weights-w_prev)))/X_row_size;
    end
    %fprintf('Number of iterations: %d\n',ii-1);
    %fprintf('tol = %e\n',tol);
elseif mode == "IRLS"
    ii = 1;
    tol = inf;
    while ii <= maxiter && tol > epsilon
        r = sigmoid(data*weights).*(1-sigmoid(data*weights));
        a = data.'*diag(r)*data;
        b = sigmoid(data*weights)-labels;
        w_prev  = weights;
        z = data*w_prev - b./r;
        %----------------------- Uncomment to use PCG ---------------------
        [weights,~] = pcg(a , data.'*(r.*z) , 1e-13 , 1000 , diag(diag(a)));
        %------------------------------------------------------------------
        %----------------------- Uncomment to use direct solver -----------
        %p = diag(diag(a));
        %weights = (p\a)\(p\(data.'*(r.*z)));
        %------------------------------------------------------------------
        ii = ii + 1;
        tol = sum(abs(data*(weights-w_prev)))/X_row_size;
    end
    %fprintf('Number of iterations: %d\n',ii-1);
    %fprintf('tol = %e\n',tol);
end

end

function sigmoid_X = sigmoid(X)

sigmoid_X = 1./(1 + exp(-X));

end