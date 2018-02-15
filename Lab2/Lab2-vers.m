%% Least-square
clc; close all; clear;

units = 6;

xTrain = (0:0.1:2*pi)';
xTest = (0.05:0.1:2*pi)';

sinTrainTarget = sin(2*xTrain);
squareTrainTarget = square(2*xTrain);

sinTestTarget = sin(2*xTest);
squareTestTarget = square(2*xTest);

target = squareTrainTarget;
testTarget = squareTestTarget;

% Initialize mu
mu = zeros(units,1);
counter=1;
segment = idivide(numel(xTrain),int32(units),'floor');
sum=0;
for i = 1:numel(xTrain)
    sum = sum + xTrain(i);
    if(mod(i,segment) == 0)
        mu(counter) = sum/double(segment);
        counter = counter + 1;
        sum = 0;
    end 
end

var = 0.5;
sDev = sqrt(var);

Phi=zeros(numel(xTrain),units);
for i = 1:numel(xTrain)
    x=xTrain(i);
    for j = 1:units
        Phi(i,j) = (exp(-(x-mu(j)).^2/(2*sDev^2)));
    end
end

W = pinv(Phi)*target;
output = sign(Phi*W);

error = abs(target-output);
avgError = mean(error);
disp("Mean error: "+ avgError);

% Test on hold-out set
PhiTest=zeros(numel(xTest),units);
for i = 1:numel(xTest)
    x=xTest(i);
    for j = 1:units
        PhiTest(i,j) = (exp(-(x-mu(j)).^2/(2*sDev^2)));
    end
end
output = PhiTest*W;
error = abs(testTarget-output);
avgError = mean(error);
disp("Mean error on hold-out set: "+ avgError);


%% Gaussian RBF Batchlearning Delta rule
clc; close all; clear;

units = 6;

xTrain = (0:0.1:2*pi)';
xTest = (0.05:0.1:2*pi)';
sinTrainTarget = sin(2*xTrain);
squareTrainTarget = square(2*xTrain);

sinTestTarget = sin(2*xTest);
squareTestTarget = square(2*xTest);

target = sinTrainTarget;
testTarget = sinTestTarget;

% Initialize mu
mu = zeros(units,1);
counter=1;
segment = idivide(numel(xTrain),int32(units),'floor');
sum=0;
for i = 1:numel(xTrain)
    sum = sum + xTrain(i);
    if(mod(i,segment) == 0)
        mu(counter) = sum/double(segment);
        counter = counter + 1;
        sum = 0;
    end 
end

W=rand(units,1);
var = 0.5;
sDev = sqrt(var);
eta = 0.0001;

epoch = 100000;
output=zeros(numel(xTrain),1);
Phi=zeros(numel(xTrain),units);

for n = 1:epoch
    for i = 1:numel(xTrain)
        x=xTrain(i);
        for j = 1:units
            Phi(i,j) = (exp(-(x-mu(j)).^2/(2*sDev^2)));
        end
        
        output(i) = Phi(i,:)*W;
    end
    dW = eta*(target'-output')*Phi;
    W = W + dW';
    error = abs(target-output);
    avgError = mean(error);
    disp(avgError)
    if (avgError < 0.1)
        disp("Number of iterations: " + n);
        break;
    end
end

% Test on hold-out set
PhiTest=zeros(numel(xTest),units);
for i = 1:numel(xTest)
    x=xTest(i);
    for j = 1:units
        PhiTest(i,j) = (exp(-(x-mu(j)).^2/(2*sDev^2)));
    end
end
output = PhiTest*W;
error = abs(testTarget-output);
avgError = mean(error);
disp("Mean error on hold-out set: "+ avgError);

%% Gaussian RBF sequential learning Delta rule
clc; close all; clear;

units = 15;

xTrain = (0:0.1:2*pi)';
xTest = (0.05:0.1:2*pi)';
sinTrainTarget = sin(2*xTrain);
squareTrainTarget = square(2*xTrain);

sinTestTarget = sin(2*xTest);
squareTestTarget = square(2*xTest);

target = sinTrainTarget;
testTarget = sinTestTarget;

% Initialize mu
mu = zeros(units,1);
counter=1;
segment = idivide(numel(xTrain),int32(units),'floor');
sum=0;
for i = 1:numel(xTrain)
    sum = sum + xTrain(i);
    if(mod(i,segment) == 0)
        mu(counter) = sum/double(segment);
        counter = counter + 1;
        sum = 0;
    end 
end

W=rand(units,1);
var = 0.5;
sDev = sqrt(var);
eta = 0.0001;

epoch = 10000;
output=zeros(numel(xTrain),1);
Phi=zeros(numel(xTrain),units);

for n = 1:epoch
    for i = 1:numel(xTrain)
        x=xTrain(i);
        for j = 1:units
            Phi(i,j) = (exp(-(x-mu(j)).^2/(2*sDev^2)));
        end
        
        output(i) = Phi(i,:)*W;
        dW = eta*(target(i)-output(i))*Phi(i,:);
        W = W + dW';
    end
    error = abs(target-output);
    avgError = mean(error);
    disp(avgError)
    if (avgError < 0.1)
        disp("Number of iterations: " + n);
        break;
    end
end

% Test on hold-out set
PhiTest=zeros(numel(xTest),units);
for i = 1:numel(xTest)
    x=xTest(i);
    for j = 1:units
        PhiTest(i,j) = (exp(-(x-mu(j)).^2/(2*sDev^2)));
    end
end
output = PhiTest*W;
error = abs(testTarget-output);
avgError = mean(error);
disp("Mean error on hold-out set: "+ avgError)




