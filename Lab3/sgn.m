function out = sgn(val)
    out = sign(val);
    if (out==0)
        out = 1;
    end
end