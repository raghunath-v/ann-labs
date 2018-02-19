function out = rbf(x,mu,sigma)
    out = exp(-(x-mu^2)/(2*sigma^2));
end