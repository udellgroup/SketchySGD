function [U,S] = NystromGLMSketch(X,R,k,p)
d = size(X,2); Omega = orth(randn(d,k+p));
Y = X'*(R.*(X*Omega));
nu = sqrt(d)*eps(norm(Y,2));
Y = Y+nu*Omega;
B = Omega'*Y;
C = chol(B);
M = Y/C;
[U,S,~] = svds(M,k+p);
S = diag(S);
S = max(S.^2-nu,0);

end

