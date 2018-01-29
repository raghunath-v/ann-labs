% 3.1 Assignment

% 3.1.1
rng default

% class 1
mu_1 = [3, 3];
sigma_1 = [1, -1.5; -1.5, 10];
c_1 = mvnrnd(mu_1, sigma_1, 100);
c_1_size = size(c_1);
% c_1_size(2); % 2
c1_label = transpose(ones(1, c_1_size(1)));
c_1 = [c_1 c1_label]; % the third column is the label
c_1(2, 2);           % (x_index, y_index)

% class 2
mu_2 = [-3, -3];
sigma_2 = [1, 1.5; 1.5, 10];
c_2 = mvnrnd(mu_2, sigma_2, 100);
c_2_size = size(c_2);
c2_label = -1*transpose(ones(1, c_2_size(1)));
c_2 = [c_2 c2_label];

% shuffle data into one dataset
data = [c_1; c_2];
shuffledData = data(randperm(size(data,1)), :);


% 3.1.2
% How quickly do the algorithms converge? 
% Plot the learning curves for each variant of learning
% Visualise the learning process by plotting a separating line (decision
% boundary) after each epoch of training

% TODO, make trainingdata integer.


trainProcent = 0.7;
shuffleSize = size(shuffledData);
trainingDataSize = shuffleSize(1)*trainProcent;

trainingData_label = shuffledData(1:trainingDataSize,:);
testData = shuffledData((trainingDataSize+1):shuffleSize(1),:);

% 1. Select random sample from training set as input
trainData = trainingData_label(:,1:2);
trainData = [ones(trainingDataSize,1) trainingData_label(:,1:2)];
trainLabel = trainingData_label(:,3:3);

% 2. If classification is correct, do nothing
% the dimension is: W*x = y      1x2 * 2x140 = 1x140 

% initialize the values
w = zeros(1, 3);
delta_w = zeros(1, 3);
output = 0;
eta = 0.00001;
thredshold = 0;
epochs = 10;
% for the sequential


for epoch = 1:epochs
    for y = 1:size(trainLabel)
        % if result is 0 and label is 1
        output = w*trainData(y,:)';
        w = w+delta_w;

        if (output < thredshold) && (trainLabel(y)==1)
            delta_w = eta*trainData(y,:);

        end

        if (output >= thredshold) && (trainLabel(y)==-1)
            delta_w = -eta*trainData(y,:);
        end
    end
    w = w/140
    
end
w


hold all
plot(c_1(:, 1), c_1(:,2), 'o');
plot(c_2(:, 1), c_2(:,2), '+');
x = -10:10;
y = w(1)/w(3) - w(2)/w(3)*x;
plot(x,y, 'r')
%plot(w(2), w(3), 'r');





