%% Section 3.1 
% RBF Batchlearning
clc; close all; clear all;

%genData;
load('data')

%plot( xTrain, squareTrainTarget);
var = 0.01;
sDev = sqrt(var);

MAX_ERROR = 0.001;
MAX_UNITS = 63;
% Check without noise
for units = 1:MAX_UNITS
% RBF parameters
    xTrainMat = repmat(xTrain,[1,units]);
    xTestMat = repmat(xTest,[1,units]);
    mean = (0:(2*pi/(units-1)):2*pi);
    muMat = repmat(mean, [train_size, 1]);
    sDevMat = repmat(sqrt(rand(1,units)), [train_size, 1]);
    Phi = exp((-(xTrainMat-muMat).^2)./(2*sDev.^2));
    newW = (inv(Phi'*Phi))*Phi'*sinTrainTarget;
    error = sum((Phi*newW-sinTrainTarget).^2)/length(sinTrainTarget);
    if (error < MAX_ERROR)
        break;
    end
end
disp('Number of units')
disp(units)
% Check with noise
for units = 1:MAX_UNITS
% RBF parameters
    xTrainMat = repmat(xTrain,[1,units]);
    xTestMat = repmat(xTest,[1,units]);
    mean = (0:(2*pi/(units-1)):2*pi);
    muMat = repmat(mean, [train_size, 1]);
    sDevMat = repmat(sqrt(rand(1,units)), [train_size, 1]);
    Phi = exp((-(xTrainMat-muMat).^2)./(2*sDev.^2));
    newW = (inv(Phi'*Phi))*Phi'*sinTrainTarget_noisy;
    newPhi = exp((-(xTestMat-muMat).^2)./(2*sDev.^2));
    error = sum((Phi*newW-sinTrainTarget_noisy).^2)/length(sinTrainTarget_noisy);
    if (error < MAX_ERROR)
        break;
    end
end
disp('Number of epochs')
disp(units)



%% Section 3.2 
% Sequential Learning and Perceptron
clc; close all; clear all;
load('data')

units = 53;
mean = (0:(2*pi/(units-1)):2*pi);
muMat = repmat(mean, [train_size, 1]);
%sDevMat = repmat(sqrt(rand(1,units)), [train_size, 1]);
var = 0.01;
sDev = sqrt(var);
xTrainMat = repmat(xTrain,[1,units]);
xTestMat = repmat(xTest,[1,units]);

W = zeros(units,1);
eta = 0.01; %Learning rate
MAX_ERRROR = 0.1;
MAX_EPOCHS = 100000;
%Online mode learning
for n = 1:MAX_EPOCHS
    error = 0;
    for i=1:length(xTrain)
        Phi = exp((-(xTrain(i)-mean).^2)/(2*sDev.^2));
        output = Phi*W;
        dW = -eta*(output-sinTrainTarget_noisy(i))*Phi
        W = W + dW';
        error = error + (output - sinTrainTarget_noisy(i))^2;
    end
    error = error/length(xTrain)
    if error<MAX_ERRROR
        disp('Number of epochs')
        disp(n)
        break;
    end
end

testPhi = exp((-(xTestMat-muMat).^2)./(2*sDev.^2));
plot(testPhi*W);

% Two layer perceptron
hiddenSizes = [13 13];
net = feedforwardnet(hiddenSizes);
net.trainFcn = 'traingd';
net.trainParam.show = 1;
net.trainParam.lr = 0.001;
net.trainParam.epochs = 100000;
net.trainParam.goal = 0.01;
net.divideFcn ='dividerand';
net.divideParam.trainRatio = 1;
net.divideParam.testRatio = 0;
net.divideParam.valRatio = 0;

net = train(net,xTrain',sinTrainTarget');
figure;
hold on
plot(sim(net, xTrain'))
plot(sim(net, xTest'))
hold off

%% Section 3.3
% Competitive Learning
clc; close all; clear;
load('data')

var = 0.01;
sDev = sqrt(var);

% Check without noise
units = 24;
% RBF parameters
xTrainMat = repmat(xTrain,[1,units]);
xTestMat = repmat(xTest,[1,units]);
mean = (0:(2*pi/(units-1)):2*pi);
%mean = rand(1,units)*2*pi;
mean = comp_learning(mean, xTrain');
muMat = repmat(mean, [train_size, 1]);
Phi = exp((-(xTrainMat-muMat).^2)./(2*sDev.^2));
newW = (inv(Phi'*Phi))*Phi'*sinTrainTarget;
error = sum((Phi*newW-sinTrainTarget).^2)/length(sinTrainTarget);

disp('Error')
disp(error)
% Check with noise
units = 53;
% RBF parameters
xTrainMat = repmat(xTrain,[1,units]);
xTestMat = repmat(xTest,[1,units]);
mean = (0:(2*pi/(units-1)):2*pi);
%mean = rand(1,units)*2*pi;
mean = comp_learning(mean, xTrain');
muMat = repmat(mean, [train_size, 1]);
Phi = exp((-(xTrainMat-muMat).^2)./(2*sDev.^2));
newW = (inv(Phi'*Phi))*Phi'*sinTrainTarget_noisy;
newPhi = exp((-(xTestMat-muMat).^2)./(2*sDev.^2));
error = sum((Phi*newW-sinTrainTarget_noisy).^2)/length(sinTrainTarget_noisy);

disp('Error')
disp(error)

% Ballist
load ballist.dat
load balltest.dat
units = 24;
max_angle = max(ballist(:,1));
max_vel = max(ballist(:,2));

mean = [(0:(max_angle/(units-1)):max_angle);(0:(max_vel/(units-1)):max_vel)];
mean = comp_learning(mean, ballist(:,1:2)');
%muMat = repmat(mean, [train_size, 1]);
var = 0.01;
sDev = sqrt(var);
%xTrainMat = repmat(xTrain,[1,units]);
%xTestMat = repmat(xTest,[1,units]);
xTrain = ballist(:,1:2);
trainTarget = ballist(:,3:4);
xTest = balltest(:,1:2);
testTarget = balltest(:,3:4);

W = zeros(units,1);
eta = 0.01; %Learning rate
MAX_ERRROR = 0.05;
MAX_EPOCHS = 100000;
%Online mode learning
for n = 1:MAX_EPOCHS
    error = 0;
    for i=1:size(xTrain,1)
        xTrainMat = repmat(xTrain(i,:)',[1,units]);
        Phi = exp((-(xTrainMat-mean).^2)/(2*sDev.^2));
        output = Phi*W;
        dW = -eta*(output-trainTarget(i,:)')'*Phi;
        W = W + dW';
        error = error + sum((output - trainTarget(i,:)').^2,1);
    end
    error = error/size(xTrain,1);
    if error<MAX_ERRROR
        disp('Number of epochs')
        disp(n)
        break;
    end
end
for i=1:size(xTest,1)
    xTestMat = repmat(xTest(i,:)',[1,units]);
    newPhi = exp((-(xTestMat-mean).^2)/(2*sDev.^2));
    output = newPhi*W;
    error = error + sum((output - trainTarget(i,:)').^2,1);
end
testError = error/size(xTest,1)

%testPhi = exp((-(xTestMat-muMat).^2)./(2*sDev.^2));
%plot(testPhi*W);



