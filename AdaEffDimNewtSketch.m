function [theta,nits,L,lambda_f,s] = AdaEffDimNewtSketch(X,y,theta,s,SketchType,lambda,a,b,eps,tau,tol,MaxIter)
%Trains a Logistic Regression Model Using Newton CG
[eta,nu,alpha] = GetParams(a,b,eps,tau);
[n,d] = size(X); 
f = @(theta) 1/n*sum(log(1+exp(-y.*(X*theta))))+lambda/2*norm(theta,2)^2; %Logistic Loss Function
nits = 0;  
 lambda_f = Inf;
while nits<MaxIter && lambda_f^2>tol
    nits = nits+1; 
    [g,D2] = GetGradHessian(X,y,theta,lambda,n);
    vns =  AppxNewtonStep(X,D2,g,s,d,lambda,SketchType);
    lambda_f = sqrt(g'*-vns);
    t = LineSearch(X,y,theta,g,vns,lambda,a,b,n);
    xnsk = theta+t*vns;
    if lambda_f>eta
        if f(xnsk)-f(theta)<= -nu
            theta = xnsk;
        else
            s = 2*s;
        end
    else
        [gplus,D2_plus] = GetGradHessian(X,y,xnsk,lambda,n);
        vplus = AppxNewtonStep(X,D2_plus,gplus,s,d,lambda,SketchType);
        lambda_f_xnsk = sqrt(gplus'*-vplus);
        if lambda_f_xnsk<= alpha*lambda_f
           theta = xnsk; lambda_f = lambda_f_xnsk; 
        else
           s = 2*s;
        end
    end
    fprintf('Iteration: n=%3d, || lambda_f ||^2_2 = %8.2e\n',nits,lambda_f^2)
  
end
L = ComputeLoss(X,y,theta,lambda,n);
end

function [eta,nu,alpha] = GetParams(a,b,eps,tau)
    eta = 1/8*(1-0.5*((1+eps)/(1-eps))^2-a)/((1+eps)/(1-eps))^3;
    nu = a*b*eta^2/(1+(1+eps)/(1-eps)*eta);
    alpha = sqrt((1+eps))/(1-eps)^((1+tau)/2)*(0.57+16^(tau)/15);
end

function [L] = ComputeLoss(X,y,theta,lambda,n)
    L = 1/n*sum(log(1+exp(-y.*(X*theta))))+lambda/2*norm(theta,2)^2; %Computes logistic loss function
end

function [g,D2] = GetGradHessian(X,y,theta,lambda,n)
    g = 1/n*(X'*(-y./(1+exp(y.*(X*theta)))))+lambda*theta;
    p = 1./(1+exp(-X*theta)); D2 = p.*(1-p)/n;
end

function [vns] = AppxNewtonStep(X,D2,g,s,d,lambda,SketchType)
    if SketchType == 0
        SHroot = ssrft(D2.^(1/2).*X,s);
    else
        SHroot = GaussianSketch(D2.^(1/2).*X,s);
    end
    if s<=d
        Hs = SHroot*SHroot'+lambda*eye(s);
        C = chol(Hs);
        vns = -1/lambda*(g-SHroot'*(C\(C'\(SHroot*g)))); 
    else
        Hs = SHroot'*SHroot+lambda*eye(d);
        C = chol(Hs);
        vns = -C\(C'\g);
    end
end


function [t] = LineSearch(X,y,theta,g,vns,lambda,a,b,n) %Performs Armijo LineSearch
   t = 1; Lold = ComputeLoss(X,y,theta,lambda,n);  
   L = ComputeLoss(X,y,theta+t*vns,lambda,n);
    while L>Lold+a*t*(g'*vns)
     t = b*t; L = ComputeLoss(X,y,theta+t*vns,lambda,n);
    end
end

