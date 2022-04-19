function [U,S,Err] = AdaptiveRandNystromAppx(A,mu,k0,kmax,tol)
Err = Inf; d = size(A,2); 
%if k0>=kmax/2
%    p = 50;
%end
Omega = []; Y = []; m = k0;
while Err>tol 
Omegab = orth(randn(d,m)); ATAY = A'*(A*Omegab); Omega = [Omega Omegab]; Y = [Y ATAY];
nu = sqrt(d)*eps(norm(Y,2)); 
Y = Y+nu*Omega;
B = Omega'*Y; C = chol(B);
M = Y/C;
[U,S] = svds(M,k0);
S = diag(S);
S = max(S.^2-nu,0);
Err = S(end)/mu;
m = k0;
k0 = 2*k0;
    if k0>=kmax
        k0 = k0-m;
        m = kmax-k0;
        k0 = kmax;
        Omegab = orth(randn(d,m)); ATAY = A'*(A*Omegab); Omega = [Omega Omegab]; Y = [Y ATAY];
        nu = sqrt(d)*eps(norm(Y,2)); 
        Y = Y+nu*Omega;
        B = Omega'*Y;
        C = chol(B);
        Y = Y/C;
        [U,S] = svds(Y,k0);
        S = diag(S);
        S = max(S.^2-nu,0);
        break
    end
end
end