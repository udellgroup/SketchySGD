function [theta,nits,L,lambda_f,Err,s] = AdaNysLogRegTrain(X,y,theta,s,smax,lambda,a,b,RatTol,Tol,MaxIter)
%Trains a Logistic Regression model using NysNewton Algorithm

[n,~] = size(X); 
nits = 0;  
lambda_f = Inf;
while nits<MaxIter && lambda_f^2>Tol
    nits = nits+1; 
    [g,D2] = GetGradHessian(X,y,theta,lambda,n);
    [U,S,Err] = AdaptiveRandNystromAppx(D2.^(1/2).*X,lambda,s,smax,RatTol);
    s = length(S);
    vns = AppxNewtonStep(U,S,lambda,g);
    lambda_f = sqrt(g'*-vns);
    t =  LineSearch(X,y,theta,g,vns,lambda,a,b,n);
    theta = theta+t*vns;
    
    fprintf('Iteration: n=%3d, || lambda_f ||^2_2 = %8.2e\n',nits,lambda_f^2)
  
end

L = ComputeLoss(X,y,theta,lambda,n);
end


function [L] = ComputeLoss(X,y,theta,lambda,n)
    L = 1/n*sum(log(1+exp(-y.*(X*theta))))+lambda/2*norm(theta,2)^2; %Computes logistic loss function
end

function [g,D2] = GetGradHessian(X,y,theta,lambda,n)
    g = 1/n*(X'*(-y./(1+exp(y.*(X*theta)))))+lambda*theta;
    p = 1./(1+exp(-X*theta)); D2 = p.*(1-p)/n;
end

function [t] = LineSearch(X,y,theta,g,vns,lambda,a,b,n) %Performs Armijo LineSearch
   t = 1; Lold = ComputeLoss(X,y,theta,lambda,n);  
   L = ComputeLoss(X,y,theta+t*vns,lambda,n);
    while L>Lold+a*t*(g'*vns)
     t = b*t; L = ComputeLoss(X,y,theta+t*vns,lambda,n);
    end
end

function [vns] = AppxNewtonStep(U,S,lambda,g)
    vns = -(U*((S+lambda).^(-1).*(U'*g))+1/lambda*(g-(U*(U'*g))));
end