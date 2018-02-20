%% Gaussian RBF Batchlearning
clc; close all; clear;

units = 1000;


xTrain = 0:0.1:2*pi;
xTest = 0.05:0.1:2*pi;

sinTrainTarget = sin(2*xTrain);
squareTrainTarget = square(2*xTrain);
train_size=length(sinTrainTarget);

sinTestTarget = sin(2*xTest);
squareTestTarget = square(2*xTest);
%plot( xTrain, squareTrainTarget);
Phi = zeros(1,units);
W = rand(1,units)';
mu = 0.5;
var = 0.5;
sDev = sqrt(var);
eta = 0.001;
max_error = 0.1;

epoch = 10000;

xTrainMat = repmat(xTrain,[units,1])';
muMat = repmat(rand(1,units),[train_size, 1]);
%sDev = sqrt(0.5)*ones(size(xTrainMat));

Phi = exp(-(xTrainMat-muMat).^2/(2*sDev^2));
%output = sum(Phi'*W, 2);


%oldW = zeros(1,units);
for n = 1:epoch
    %output = zeros(1,units);
    %Phi = exp(-(xTrain-mu).^2/(2*sDev^2));
    output = Phi*W;
    dW = -eta*(output-sinTrainTarget')'*Phi;
    W = W + dW';
    V = (output - sinTrainTarget').^2;
    error = sum(V)/length(V)
    if (error < max_error)
        disp('Number of iterations:');
        disp(n);
        break;
    end
    %plot(output)
    %pause
    oldW = W;
end

%% Gaussian RBF Incremental
clc; close all; clear;

units = 5;

xTrain = 0:0.1:2*pi;
xTest = 0.05:0.1:2*pi;

sinTrainTarget = sin(2*xTrain);
squareTrainTarget = square(2*xTrain);

sinTestTarget = sin(2*xTest);
squareTestTarget = square(2*xTest);
%plot( xTrain, squareTrainTarget);
Phi = zeros(1,units);
W = rand(1,units);
mu = 0.5;
var = 0.5;
sDev = sqrt(var);
eta = 0.0001;

epoch = 500000;

Phi = exp(-(xTrain-mu).^2/(2*sDev^2));
%output = sum(Phi'*W, 2);
oldW = zeros(1,units);

for n = 1:epoch
    outputs = zeros(1,units);
    for i = 1:size(xTrain,1)
        Phi = exp(-(xTrain(i)-mu).^2/(2*sDev^2));
        output = sum(Phi*W, 2);
        dW = -eta*(output-sinTrainTarget(i))*Phi;
        W = W + dW;
    end
    outputs = sum(Phi'*W, 2);
    V = outputs - sinTrainTarget;
    error = sum(V)/length(V);
    if (error < 0.001)
        disp('Number of iterations:');
        disp(n);
        break;
    end
    oldW = W;
end



%% wtf is this
x = xTrain';
targets = sinTrainTarget;

% Inititialisation of RBF units with fixed positions over the data set.
% m	: vector of RBF positions
% var: vector of RBF variances

margin=0.1;		% How large margin outside data set

fmin=min(min(x));	% min for all dimensions
fmax=max(max(x));	% max for all dimensions

xmin=fmin-(fmax-fmin)*margin;
xmax=fmax+(fmax-fmin)*margin;

% Initialises a set of RBFs to be equidistantly positioned over the ONE
% DIMENSIONAL space with fixed sized variances.
% A correction has been made in the line below. The actual number of units
% was one more than specified by units.
dist = ((fmin-fmax)/(units-1));
%m is a column vector with positions (for several dimension => more columns)
m = (fmin:dist:fmax)';
% Let them also have the same standard deviation
sdev = m*0 +0.5 * dist;
var = sdev.*sdev;

%Initialize the weight vector with random values
w = rand(rows(m),1)+0.1;

%These variables are used by lmsiter.m
iter=0;
itersub=20;		% subsampling
itermax=2000;		% # iterations per call to diter (multiple of itersub)

delta=0.2;		% used by others

Phi=zeros(rows(x),rows(m));
%Calculate the RBF responses
for i=1:rows(x)
  gauss=exp(-sqdist(m,x)./(2*pi*(var)));
  gauss=gauss/sum(gauss);
  Phi(i,:)=gauss';
end

W = Phi\targets; %least square solution

y = sign(Phi*W);
% 
% rbfplot1(patterns,y,targets,units);

% subplot(2,1,1); plot(xTrain,y,xTrain',yd);
% title([' Function y and desired y, RBF-units=' int2str(units)]);
% subplot(2,1,2); plot(x,yd-y);
% title(['Residual, max= ' num2str(max(abs(yd-y)))]);
% 


% figure;
% hold;
% 
% for unitI=1:numberOfUnits,
%     plot(patterns,W(unitI)*Phi(:,unitI));
%     %pause();
% end
% title(['Transfer functions for RBFs using ', int2str(numberOfUnits), ' units']);
% %W