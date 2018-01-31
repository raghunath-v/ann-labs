function [] = genData8(N_DATA_PER_CLASS)
    %N_DATA_PER_CLASS=125;
    N_FEATURES=8;
    % 3.1.1
    rng default

    %Generate Data
    % class 1
    c_1 = [1 -1 -1 -1 -1 -1 -1 -1; -1 1 -1 -1 -1 -1 -1 -1; -1 -1 1 -1 -1 -1 -1 -1; -1 -1 -1 -1 1 -1 -1 -1];

    c1_label = ones(size(c_1,1), 1);

    c_1 = [c_1 c1_label]; % the third column is the label


    % class 2
    c_2 = [-1 -1 -1 1 -1 -1 -1 -1; -1 -1 -1 -1 -1 1 -1 -1; -1 -1 -1 -1 -1 -1 1 -1; -1 -1 -1 -1 -1 -1 -1 1];

    c2_label = -1*(ones(size(c_2, 1), 1));
    c_2 = [c_2 c2_label]

    % shuffle data into one dataset
    data = [c_1; c_2];
    shuffledData = data(randperm(size(data,1)), :);
   
    figure();
    hold on
    plot(c_1(:, 1), c_1(:,2), 'o');
    plot(c_2(:, 1), c_2(:,2), '+');
    hold off
    
    save('data8.mat')
end