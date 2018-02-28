clear all; clc; close all

N = 8; % Number of units
P = 256; % Number of patterns
%X = 2*double(rand(P, N)>0.5)-1;
X = 2*(dec2bin(2^N-1:-1:0)-'0')-1;
%X = X(randperm(size(X,1)),:);

out = zeros(256,1);
for i = 1:256
    out(i)=isAttractor(X(i,:));
end