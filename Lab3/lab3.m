%% Section 3.1
% Hopfield network
clear all; clc; close all

N = 8; % Number of units
P = 3; % Number of patterns
X = 2*(dec2bin(2^N-1:-1:0)-'0')-1;
X = X(randperm(size(X,1)),:);
W = zeros(N);

% Inputs
x1=[-1 -1 1 -1 1 -1 -1 1];
x2=[-1 -1 -1 -1 -1 1 -1 -1];
x3=[-1 1 1 -1 -1 1 -1 1];
Xt = [x1;x2;x3];

% Distorted inputs
x1d=[ 1 -1 1 -1 1 -1 -1 1];
x2d=[ 1 1 -1 -1 -1 1 -1 -1];
x3d=[ 1 1 1 -1 1 1 -1 1];
Xd = [x1d;x2d;x3d];

Xtemp = X;

W=Xt'*Xt/N;
%for i=1:P
%    W = W + Xt(i,:)'*Xt(i,:);
%end

disp('Recalling the inputs from distorted values. Errors: ')
disp(sgn(Xt*W)-sgn(Xd*W));

for i=1:100
    oldXtemp = Xtemp;
    Xtemp = sgn(Xtemp*W);
    err = sum(sum(abs(Xtemp - oldXtemp)))
    if (err==0)
        disp('Number of iterations:')
        disp(i);
        break;
    end
end

X_attr = 2*(dec2bin(2^N-1:-1:0)-'0')-1;



%% Section 3.2
clear all; clc; close all

load pict.dat
X = reshape(pict,1024,length(pict)/1024)';
Xt = X(1:3,:);
Xtest = 2*double(rand(9,1024)>0.5)-1;

N = 1024; % Number of units
P = 9; % Number of patterns
%W = zeros(N);

W=Xt'*Xt/N;

% figure;
% subplot(1,2,1)
% imshow(reshape(X(1,:),32,32)')
% title('p1')
% subplot(1,2,2)
% test = sgn(X(10,:)*W);
% imshow(reshape(test,32,32)')
% title('p10 reconstructed')
% 
% figure;
% subplot(1,3,1)
% imshow(reshape(X(2,:),32,32)')
% title('p2')
% subplot(1,3,2)
% imshow(reshape(X(3,:),32,32)')
% title('p3')
% subplot(1,3,3)
% test = sgn(X(11,:)*W);
% imshow(reshape(test,32,32)')
% title('p11 reconstructed')

% figure;
% for j=1:9
%     subplot(3,3,j)
%     img = reshape(X(j,:),32,32)';
%     imshow(img);
%     title(strcat('p',num2str(j)))
% end

%Xtemp = X(1:9,:);
%Xtemp = 2*double(rand(1,N)>0.5)-1;
Xtemp = X(11,:);
%Xtemp = X(8,:);
figure;
for i=1:1000
    oldXtemp = Xtemp;
    %energy(W,Xtemp)
    for j=1:1024
        Xtemp(j) = sgn(Xtemp*W(:,j));
        if (mod(j,100)==0)
            img = reshape(Xtemp(1,:),32,32)';
            imshow(img);
            pause(0.1)
        end
    end
    err = sum(sum(abs(Xtemp - oldXtemp)))
    if (err==0)
        disp('Number of iterations:')
        disp(i);
        break;
    end
end



%% Section 3.3
% Energy
clear all; clc; close all

load pict.dat
X = reshape(pict,1024,length(pict)/1024)';
Xt = X(1:3,:);
Xtest = 2*double(rand(9,1024)>0.5)-1;

N = 1024; % Number of units
P = 9; % Number of patterns
%W = zeros(N);
Xtemp = X(1:9,:);

W=Xt'*Xt/N;
%W=2*rand(N)-1;
%W =(W+W')/2;

energy(W,Xtemp)

pause
Xtemp = 2*double(rand(1,N)>0.5)-1;
%Xtemp = X(8,:);
figure;
for i=1:1000
    oldXtemp = Xtemp;
    energy(W,Xtemp)
    for j=1:1024
        Xtemp(j) = sgn(Xtemp*W(:,j));
        if (mod(j,10)==0)
            img = reshape(Xtemp(1,:),32,32)';
            imshow(img);
            pause(0.1)
        end
    end
    err = sum(sum(abs(Xtemp - oldXtemp)));
    if (err==0)
        disp('Number of iterations:')
        disp(i);
        break;
    end
end

%% Section 3.4 Distortion Resistance

clear all; clc; close all

load pict.dat
X = reshape(pict,1024,length(pict)/1024)';
Xt = X(1:3,:);
Xtest = 2*double(rand(9,1024)>0.5)-1;

N = 1024; % Number of units
P = 9; % Number of patterns
Xtemp = X(1:9,:);

W=Xt'*Xt/N;

ratio = 0.2;
limit = ceil(N*ratio);
noise = [ones(1,N-limit), -ones(1,limit)]; 
noise = noise(randperm(N));
Xtest = X(2,:).*noise;

figure
subplot(1,2,1)
img = reshape(X(2,:),32,32)';
imshow(img);
subplot(1,2,2)
img = reshape(sgn(Xtest*W),32,32)';
imshow(img);

noMatch=zeros(3,1000);
for i = 1:1000
    noise = [ones(1,N-limit), -ones(1,limit)]; 
    noise = noise(randperm(N));
    for j = 1:3
        Xtest = X(j,:).*noise;
        noMatch(j,i) = double(sum(sgn(Xtest*W)-X(j,:))>0);
    end
end

sum(noMatch,2)

%% Section 3.5 Capacity Part 1
clear all; clc; close all

load pict.dat
N = 1024; % Number of units
P = 9; % Number of patterns
TrainP = 9; % Number of patterns to train
X = reshape(pict,N,length(pict)/N)';   % Original data
X = 2*double(rand(9,1024)>0.5)-1;    % Data with random patterns
Xt = X(1:TrainP,:);
Xtemp = X(1:9,:);

W=Xt'*Xt/N;

ratio = 0.3;
limit = ceil(N*ratio);
noise = [ones(1,N-limit), -ones(1,limit)]; 
noise = noise(randperm(N));
Xtest = X(2,:).*noise;

figure
subplot(1,2,1)
img = reshape(X(2,:),32,32)';
imshow(img);
subplot(1,2,2)
img = reshape(sgn(Xtest*W),32,32)';
imshow(img);

noMatch=zeros(TrainP,1000);
for i = 1:1000
    noise = [ones(1,N-limit), -ones(1,limit)]; 
    noise = noise(randperm(N));
    for j = 1:TrainP
    Xtest = X(j,:).*noise;
    noMatch(j,i) = double(sum(sgn(Xtest*W)-X(j,:))>0);
    end
end

sum(noMatch,2)


%% Section 3.5 Capacity Part 2
clear all; clc; close all

N = 100; % Number of units
P = 300; % Number of patterns
%TrainP = 9; % Number of patterns to train
%X = reshape(pict,N,length(pict)/N)';   % Original data
%X = 2*double(rand(P,N)>0.5)-1;    % Data with random patterns
bias = 0;
X = sgn(bias+randn(P,N)); 

ratio = 0.2;
limit = ceil(N*ratio);
noise = [ones(1,N-limit), -ones(1,limit)]; 
noise = noise(randperm(N));

numErr = [];
for trainP = 1:P
    Xt = X(1:trainP,:);
    W=Xt'*Xt/N;
    W=W-diag(diag(W));
    noMatch=zeros(trainP,1);
    for j = 1:trainP
        noise = [ones(1,N-limit), -ones(1,limit)]; 
        noise = noise(randperm(N));
        Xtest = X(j,:).*noise;
        noMatch(j) = double(sum(sgn(Xtest*W)-X(j,:))>0);
    end
    numErr = [numErr sum(noMatch)];
end
plot(numErr)

%% Section 3.6 Sparse Patterns
clear all; clc; close all

sparsity = 0.01;
bias = 0.5;
N=100;
P=300;
X = [ones(P,ceil(N*sparsity)), zeros(P,N-ceil(N*sparsity))]; 
max_epochs = 100;
ratio = 0;
limit = ceil(N*ratio);
%X = X(randperm(N));
for c = 1:P
    X(c,randperm(N)) = X(c,:);
end
W=getWeight(X,sparsity);

numErr = [];
for trainP = 1:P
    Xt = X(1:trainP,:);
    W=getWeight(Xt,sparsity);
    %W=W-diag(diag(W));
    noMatch=zeros(trainP,1);
    for j = 1:trainP
        Xtest = X(j,:);
%         for i=1:max_epochs
%             oldXtest = Xtest;
%             Xtest = updateXbias(Xtest);
%             err = sum(sum(abs(Xtest - oldXtest)));
%             if (err==0)
%                 break;
%             end
%         end
        noMatch(j) = double(sum(updateXbias(Xtest,W,bias)-X(j,:))>0);
    end
    numErr = [numErr sum(noMatch)];
end
plot(numErr)
title('bias=0.01')
