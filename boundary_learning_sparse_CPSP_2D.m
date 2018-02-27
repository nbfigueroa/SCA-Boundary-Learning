%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SVM_perf TEST.. specifically CPSP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 2D Robot Collision Data
clear all; close all; clc;
load('./2d-example2.mat')
X_train = X; y_train = labels;

%% 36D Real Robot Collision Data
clear all; close all;

% KUKA Innovation Award Setup - with sample_col = 2
% dataset_name = '../function_learning/collisionDatasets/data_mat/Innovation_Award_Dataset.mat';

% New LASA lab Dual-Arm IIWA setup (Feb 2018) - with sample_col = 1 - 40 deg resolution
dataset_name = '../function_learning/collisionDatasets/data_mat/New_IIWA_Setup_Feb18_Dataset.mat';
load(dataset_name)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Partition Dataset into Train+Validation/Test %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt_ratio = 0.3;
[ X_train, y_train, X_valid, y_valid ] = split_data(X', y', tt_ratio );

X_train = X_train'; y_train = y_train';
X_valid  = X_valid';  y_valid  = y_valid';

%% LibSVM Performance
totSV = model.totalSV;
ratioSV = totSV/length(y_train);
posSV = model.nSV(1)/totSV;
negSV = model.nSV(2)/totSV;
boundSV = sum(abs(model.sv_coef) == options.C)/totSV;

fprintf('*SVM Model Statistic*\n Total SVs: %d, SV/M: %1.4f \n +1 SV : %1.4f, -1 SVs: %1.4f, Bounded SVs: %1.4f \n', ...
    totSV, ratioSV, posSV, negSV,  boundSV);

% Evaluate SVM performance on Testing Set
testSamples = 100000;

% Test Learnt Model
[y_est] = svm_classifier(X_test(1:testSamples,:), y_test(1:testSamples,:), [], model);

% Compute Classifier Test Stats
[test_stats] = class_performance(y_test(1:testSamples,:),y_est); 
fprintf('*Classifier Performance on Test Set (%d points)* \n Acc: %1.5f, F-1: %1.5f, FPR: %1.5f, TPR: %1.5f \n', ...
    testSamples, test_stats.ACC, test_stats.F1, test_stats.FPR, test_stats.TPR)

%% Prepare Data for CPSP (SVM_perf)
options.C     = 2222;
options.sigma = 0.7333;
testSamples = 100000;

clc;
C_perf = ((options.C)/100)*length(y_train);
fprintf('C_perf: %10.1f\n',C_perf)
Gamma = 1/(2*options.sigma^2)
svmlwrite('./svm_perf/robotCollision/36d-811k-Collision-Fender-Points.dat', X_train, y_train)
svmlwrite('./svm_perf/robotCollision/36d-811k-Collision-Fender-Testing.dat', X_test(1:testSamples,:), y_test(1:testSamples,:))

%% Train sparse SVM via CSPS
% For now this is done in the command line.. I will change it later to a
% more matlab-friendly way

% Command to train and test CPSP algo
% Execute the following command in terminal to learn the Sparse SVM
% ./svm_perf_learn -c 100 -t 2 -g 50 --i 2 -w 9 --b 0 --k 100 ./robotCollision/2d_example.dat ./robotCollision/2d_model.dat

% Execute the following command in terminal to test the learnt SVM
% ./svm_perf_classify ./robotCollision/36d-13k-Collision-Fender-Testing.dat ./robotCollision/36d-13k-Collision-Fender-Model.dat ./robotCollision/36d-13k-Collision-Fender-Predictions.dat

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test Performance of Sparse SVM learnt via CPSP %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Load Testing Dataset
clear all; clc
foldername = './svm_perf/robotCollision/';
[y_test, X_test] = svmlread(strcat(foldername,'36d-130k-Collision-Fender-Testing.dat'));

%% Load Predictions
foldername = './svm_perf/robotCollision/';
prediction_names = dir('./svm_perf/robotCollision/36d-*k-Collision-Fender-Predictions-3000.dat'); 
nPredictions = length(prediction_names);
predictions  = zeros(length(y_test),nPredictions);
for i=1:nPredictions
    predictions(:,i) = sign(svmlread(strcat(foldername,prediction_names(i).name)));
end

% Dataset Sizes
M = [ 67000 130000 270000 540000 ];

%% Compute Error Rates for each model on test set
ACC = zeros(1,nPredictions);
F1  = zeros(1,nPredictions);
FPR = zeros(1,nPredictions);
TPR = zeros(1,nPredictions);
clc;
for i =  1:nPredictions    
    % Compute Classifier Test Stats
    [stats] = class_performance(y_test,predictions(:,i));
    fprintf('*Classifier Performance for M=%d, k=3000 on Test Set (%d points)* \n Acc: %1.5f, F-1: %1.5f, FPR: %1.5f, TPR: %1.5f \n', M(i), length(y_test), stats.ACC, stats.F1, stats.FPR, stats.TPR)
    ACC(1,i) = stats.ACC; F1(1,i) = stats.F1;
    FPR(1,i) = stats.FPR; TPR(1,i) = stats.TPR;
    
end

%% Plot of model performance stats on test set
figure('Color',[1 1 1])
train_sizes = M;    
plot(train_sizes, ACC, '--oc','LineWidth',2,'MarkerFaceColor','c'); hold on;
plot(train_sizes, F1,  '--vk','LineWidth',2,'MarkerFaceColor','k'); hold on;
plot(train_sizes, 1-FPR, '--dr','LineWidth',2,'MarkerFaceColor','r'); hold on;
plot(train_sizes, TPR, '--sm','LineWidth',2,'MarkerFaceColor','m'); hold on;
ylim([0.94 1])
set(gca,'xscale','log')
xlim([60000 550000])

xlabel('Training Set Size','Interpreter','Latex','FontSize',20);
ylabel('Performance Measure','Interpreter','Latex','FontSize',20)
title('Sparse SVM Models learned with $k_{sv}=3000$','Interpreter','Latex', 'FontSize',18)
legend({'ACC','F1','1-FPR','TPR'},'Interpreter','Latex', 'FontSize',14)
grid on

%% Load Choosen Model (540k k=3000 SVs)
folder_name   = './svm_perf/robotCollision/';
% optimal_model_name = '36d-067k-Collision-Fender-Model-3000.dat';
optimal_model_name = '2d_model.dat';
cpsp_model_2dexample = readCPSPmodel(strcat(folder_name,optimal_model_name));



