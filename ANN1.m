% 3.1 Assignment

% 3.1.1
rng default

% class 1
mu_1 = [5, 5];
sigma_1 = [1, -1.5; -1.5, 10];
c_1 = mvnrnd(mu_1, sigma_1, 100);
c_1_size = size(c_1);
% c_1_size(2); % 2
c1_label = transpose(ones(1, c_1_size(1)));
c_1 = [c_1 c1_label]; % the third column is the label
c_1(2, 2);           % (x_index, y_index)

% class 2
mu_2 = [-5, -5];
sigma_2 = [1, 1.5; 1.5, 10];
c_2 = mvnrnd(mu_2, sigma_2, 100);
c_2_size = size(c_2);
c2_label = -1*transpose(ones(1, c_2_size(1)));
c_2 = [c_2 c2_label];

% shuffle data into one dataset
data = [c_1; c_2];
shuffledData = data(randperm(size(data,1)), :);

hold all
plot(c_1(:, 1), c_1(:,2), 'o');
plot(c_2(:, 1), c_2(:,2), '+');

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
trainLabel = trainingData_label(:,3:3);

% 2. If classification is correct, do nothing
eta = 0.1;
delta_w = eta*trainData;
% a = size(trainData)
thredshold = [0, 0];
% the dimension is: W*x = y      2x2 * 2x140 = 2x140 
w = zeros(2, 2);
output = w*transpose(trainData);
%size(result)
output = transpose(output);
size_output = size(output);
size_output(1)          % 140
size_output(2)          % 2
%x = 0

% for the sequential
for y = 1:size(trainLabel)
    
    for param = 1:size_output(2)
        % if the output is higher than thredshold and y is 
       % if (output(y,param) >= 0) && (trainLabel(y)==-1)
        %    delta_w = -eta*train;
           
            
      %  end 
    end
    %if result(y)<thredshold
    %if result(y,:) < thredshold
    % end
    
end
%x

% a = output(1,:) < thredshold

output(1,:)
trainLabel(3)

%{
predict = act_f(y)
if (predict == label(index))
    % do nothing
end

else 
%}
