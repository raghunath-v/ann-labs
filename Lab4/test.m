clear all; clc; close all;
load binMNIST

traindata = bindata_tst;
testData = bindata_trn;

N = 8000;
V = 784;
H = 150;
nodes = [V H];
rbm = randDBN(nodes);

opts.MaxIter = 100;
opts.BatchSize = 100;
opts.Verbose = true;
opts.StepRatio = 0.1;
rbm = pretrainDBN(rbm, traindata, opts);

plot();

