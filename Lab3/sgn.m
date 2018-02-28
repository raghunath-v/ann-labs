function out = sgn(val)
    out = sign(val);
    out = out+0.1;
    out = sign(out);
end