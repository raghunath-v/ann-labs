function [] = genData(N_DATA_PER_CLASS)
    %N_DATA_PER_CLASS=125;
    N_FEATURES=2;
    % 3.1.1
    rng default

    %Generate Data
    % class 1
    mu_1 = [1.5, 1.5];
    sigma_1 = [1, -1.5; -1.5, 10];
    c_1 = mvnrnd(mu_1, sigma_1, N_DATA_PER_CLASS);
    c1_label = ones(size(c_1,1), 1);
    c_1 = [c_1 c1_label]; % the third column is the label


    % class 2
    mu_2 = [-1.5, -1.5];
    sigma_2 = [1, 1.5; 1.5, 10];
    c_2 = mvnrnd(mu_2, sigma_2, N_DATA_PER_CLASS);
    c2_label = -1*(ones(size(c_2, 1), 1));
    c_2 = [c_2 c2_label];

    % shuffle data into one dataset
    data = [c_1; c_2];
    shuffledData = data(randperm(size(data,1)), :);
   
    figure();
    hold on
    plot(c_1(:, 1), c_1(:,2), 'o');
    plot(c_2(:, 1), c_2(:,2), '+');
    hold off
    
    save('data.mat')
end