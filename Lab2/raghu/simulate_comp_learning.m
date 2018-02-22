clear all; clc; close all 
load ballist.dat
load balltest.dat
units = 24;
max_angle = max(ballist(:,1));
max_vel = max(ballist(:,2));

%mean = [(0:(max_angle/(units-1)):max_angle);(0:(max_vel/(units-1)):max_vel)];
mean = [rand(1,units)*max_angle;rand(1,units)*max_vel];

mean = comp_learning(mean, ballist(:,1:2)');

