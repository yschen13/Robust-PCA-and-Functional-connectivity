function [M,L,S] = RandSL(n,r_pre,k_pre,magS)
%{
    Generate random square low rank (L) and sparse(S)
    INPUT: 
        n: square matrix  dimension
        r_pre: rank/n
        k_pre: sparsity
        magS: magnitude of S value
%}
    r = floor(r_pre*n); 
    k = floor(k_pre*n^2); % sparsity 5%
    X = randn([n,r])*1/n;
    Y = randn([n,r])*1/n;
    L = X*Y'; % low rank matrx
    A = rand(n);
    E = double(A<0.5)*2-1; % sign of S
    A = rand(n);
    O = double(A<k_pre); % support set
    S = E.* O * magS; % sparse -1/1 matrix
    M = L + S;
end
