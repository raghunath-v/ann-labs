function out = getWeight(X,rho)
    [P, N] = size(X);
    %rho = sum(sum(X))/(N*P);
    X = X-rho;
    W = X'*X/N;
    out = W;
end