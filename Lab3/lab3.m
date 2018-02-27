%% Section 3.1
clear all; clc; close all

N = 8; % Number of units
P = 3; % Number of patterns
X = 2*double(rand(P, N)>0.5)-1;

W = zeros(N);

%W=X'*X/N;

x1=[-1 -1 1 -1 1 -1 -1 1];
x2=[-1 -1 -1 -1 -1 1 -1 -1];
x3=[-1 1 1 -1 -1 1 -1 1];

X = [x1;x2;x3];

for i=1:P
    W = W + X(i,:)'*X(i,:);
end




