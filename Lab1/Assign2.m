%% Part 2
clc; close all; clear;

% Generate data
t = 301:1500;
input = [Euler(t-20); Euler(t-15);Euler(t-10);Euler(t-5);Euler(t)];
output = Euler(t+5);


% visualize the data
% Plotting time series
%{
plotOutput=Euler(301:1500);
plot(301:1500,plotOutput)
xlabel('Time','fontsize',15); ylabel('x(t)','fontsize',15);
title('Mackey-Glass time series')
%}
sd = 0.09;
noise = normrnd(0, sd^2, 5,1200);
input=input+noise;

% Feedforward
hiddenNodes = 5;
secondHiddenNodes = 4;
% can add hidden layer in hiddenSizes
hiddenSizes = [hiddenNodes secondHiddenNodes];
trainFcn = 'traingd';
net = feedforwardnet(hiddenSizes,trainFcn);

% Variables
net.trainParam.show = 1;
net.trainParam.lr = 0.005;
net.trainParam.epochs = 100000;
net.trainParam.goal = 0.05;
net.performParam.regularization = 0.05;

net.divideFcn ='divideind';

ix = randperm(1200);
ix1 = ix(1:700);
ix2 = ix(701:1000);
ix3 = ix(1001:1200);
net.divideParam.trainInd = ix1;
net.divideParam.valInd = ix2;
net.divideParam.testInd = ix3;

%net.divideind(1200,1:500,501:1000,1001:1200);
%net.trainParam.showWindow = true;
%net.permormFcn = 'mse';

% pool = parpool;
net = train(net,input,output,'useParallel','yes');
%}