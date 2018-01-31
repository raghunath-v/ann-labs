function [x] = Euler(t)
% EULER
tMax = t(length(t));
x = zeros(1,tMax+1);
x(1)=1.5;
for i=1:tMax+1
    if (i<26)
        x(i+1) = 0.9*x(i);
    else
        x(i+1)= 0.9*x(i)+ 0.2*x(i-25)/(1+x(i-25)^10);
    end
end
x = x(t);