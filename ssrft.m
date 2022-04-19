function [SA] = ssrft(A,m)
n = size(A,1); J = randperm(n,m)'; 
z = 2*binornd(1,0.5,n,1)-1; % Random signs
A = bsxfun(@times,z,A);
SA = dct(A); SA = sqrt(n/m)*SA(J,:);
end

