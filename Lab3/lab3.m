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


%% Section 3.2
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

figure;
subplot(1,2,1)
imshow(reshape(X(1,:),32,32)')
title('p1')
subplot(1,2,2)
test = sgn(X(10,:)*W);
imshow(reshape(test,32,32)')
title('p10 reconstructed')

figure;
subplot(1,3,1)
imshow(reshape(X(2,:),32,32)')
title('p2')
subplot(1,3,2)
imshow(reshape(X(3,:),32,32)')
title('p3')
subplot(1,3,3)
test = sgn(X(11,:)*W);
imshow(reshape(test,32,32)')
title('p11 reconstructed')

figure;
for j=1:9
    subplot(3,3,j)
    img = reshape(X(j,:),32,32)';
    imshow(img);
    title(strcat('p',num2str(j)))
end

figure;
for i=1:10000
    oldXtemp = Xtemp;
    Xtemp = sgn(Xtemp*W);
    err = sum(sum(abs(Xtemp - oldXtemp)));
    if (mod(i,1)==0)
        for j=1:9
            subplot(3,3,j)
            img = reshape(sgn(Xtest(j,:)*W),32,32)';
            imshow(img);
            title(strcat('p',num2str(j)))
        end
%         pause
    end
    if (err==0)
        disp('Number of iterations:')
        disp(i);
        break;
    end
end


