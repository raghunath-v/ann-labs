function out = dPhi(x)
    out = (1+Phi(x)) .* (1-Phi(-x)) / 2;
end