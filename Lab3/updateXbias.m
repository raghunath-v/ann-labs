function out = updateXbias(x,W,bias)
    n=length(x);
    for j=1:n
        x(j) = 0.5 + (0.5*sgn((x*W(:,j))-bias));
    end
    out = x;
end