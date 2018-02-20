function [] = genData()
    xTrain = (0:0.1:2*pi)';
    xTest = (0.05:0.1:2*pi)';
    sinTrainTarget = sin(2*xTrain);
    squareTrainTarget = square(2*xTrain);
    sinTestTarget = sin(2*xTest);
    squareTestTarget = square(2*xTest);
    train_size=length(sinTrainTarget);
    test_size = length(sinTestTarget);
    
    save('data.mat')
end