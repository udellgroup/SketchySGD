function [SA] = GaussianSketch(A,m)
n = size(A,1);
S = randn(m,n)/sqrt(m);
SA = S*A;
end

