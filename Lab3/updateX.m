function out = updateX(x,W)
    for j=1:length(x)
        x(j) = sgn(x*W(:,j));
    end
    out = x;
end