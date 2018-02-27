%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SVM_perf TEST.. specifically CPSP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%           STEP 1: LOAD DATASET                %
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all;

% KUKA Innovation Award Setup - with sample_col = 2
% dataset_name = '../function_learning/collisionDatasets/data_mat/Innovation_Award_Dataset.mat';

% New LASA lab Dual-Arm IIWA setup (Feb 2018) - with sample_col = 1 - 40 deg resolution
dataset_name = '../function_learning/collisionDatasets/data_mat/New_IIWA_Setup_Feb18_Dataset.mat';
load(dataset_name)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  STEP 2: Partition Dataset into Train+Validation/Test  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt_ratio = 0.1; % This can go up to 100k points!
[ X_train, y_train, X_valid, y_valid ] = split_data(X', y', tt_ratio );

X_train = X_train'; y_train = y_train';
X_valid  = X_valid';  y_valid  = y_valid';

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  STEP 3: Load Optimal Model learnt for Cross-Validation with C-SVM  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Model 
model_path = '/home/nbfigueroa/Dropbox/gaga and borat/IJRR/sparseSVMLearning/models';
model_name = '/NewSetup-IIWAs-Feb18/36D-24k-40deg-New-IIWA-Setup-OptimalModel.mat';
load(strcat(model_path, model_name))

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

% NOTE: This is the performance that the Sparse SVM model should reach!

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         STEP 4: Prepare Data for Sparse SVM Learning via CPSP       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Prepare Data for CPSP (SVM_perf)

% Optimal Model Parameters from CV with libSVM
C_opt = 16.6810;
w_opt = 0.3882;

% Options for sparse SVM training
options.C     = C_opt;
options.sigma = w_opt;
testSamples = length(X_train); % same number of test-samples.. Default:100k
options.C_perf = ((options.C)/100)*length(y_train);
options.Gamma = 1/(2*options.sigma^2);
fprintf('C_perf: %10.1f\n',options.C_perf)
fprintf('Gamma : %10.1f\n',options.Gamma)

% Generate Training files
svmlwrite('./svm_perf/robotCollision/36d-120k-NewIIWASetup-Feb18-Points.dat', X_train, y_train)
svmlwrite('./svm_perf/robotCollision/36d-120k-NewIIWASetup-Feb18--Testing.dat', X_test(1:testSamples,:), y_test(1:testSamples,:))

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%         STEP 5: Prepare Data for Sparse SVM Learning via CPSP       %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Train sparse SVM via CSPS
% For now this is done in the command line.. I will change it later to a
% more matlab-friendly way
% Command to train and test CPSP algorithm

% Execute the following command in terminal to learn the Sparse SVM
% From http://www.cs.cornell.edu/people/tj/svm_light/svm_perf.html
% -c float    -> C: trade-off between training error and margin (default 0.01)
% -t int      -> type of kernel function:
%                 0: linear (default)
%                 1: polynomial (s a*b+c)^d
%                 2: radial basis function exp(-gamma ||a-b||^2)
%                 3: sigmoid tanh(s a*b + c)
%                 4: user defined kernel from kernel.h
% -g float    -> parameter gamma in rbf kernel
% --i [0..]   -> Use CPSP algorithm for sparse kernel training
%                 (must use '-w 9') (see [Joachims/Yu/09]):
%                 0: do not use CPSP algorithm.
%                 1: CPSP using subset selection for preimages via 59/95 heuristic 
%                 2: CPSP using fixed point search (RBF kernel only) (default)
%                 4: CPSP using fixed point search with starting point via 59/95 heuristic (RBF kernel only)
% -w [0,..,9] -> choice of structural learning algorithm (default 9):
%                 0: n-slack algorithm described in [2]
%                 1: n-slack algorithm with shrinking heuristic
%                 2: 1-slack algorithm (primal) described in [5]
%                 3: 1-slack algorithm (dual) described in [5]
%                 4: 1-slack algorithm (dual) with constraint cache [5]
%                 9: custom algorithm in svm_struct_learn_custom.c
% --b float   -> value of L2-bias feature. A value of 0 implies not
%                 having a bias feature. (default 1)
%                 WARNING: This is implemented only for linear kernel!
% --k [0..]   -> Specifies the number of basis functions to use
%                 for sparse kernel approximation (both --t and --i). (default 500)

% Example for 2D Dataset
% ./svm_perf_learn -c 100 -t 2 -g 50 --i 2 -w 9 --b 0 --k 100 ./robotCollision/2d_example.dat ./robotCollision/2d_model.dat

% Example for 36D Robot Collision Dataset 
% - LEARN SPARSE MODEL WITH ALREADY
% ./svm_perf_learn -c 20761.2 -t 2 -g 3.3 --i 2 -w 9 --b 0 --k 3000 ./robotCollision/36d-120k-NewIIWASetup-Feb18-Points.dat ./robotCollision/36d-120k-NewIIWASetup-Feb18-Model.dat

% Execute the following command in terminal to test the learnt SVM
% - TEST SPARSE MODEL ON TRAINING DATA 
% ./svm_perf_classify ./robotCollision/36d-120k-NewIIWASetup-Feb18--Testing.dat ./robotCollision/36d-120k-NewIIWASetup-Feb18-Model.dat ./robotCollision/36d-120k-NewIIWASetup-Feb18-Predictions.dat

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   STEP 6: Check Performance of Sparse SVM learnt on Testing Set     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Testing Dataset
clear all; clc
foldername = './svm_perf/robotCollisposion/';
[y_test, X_test] = svmlread(strcat(foldername,'36d-120k-NewIIWASetup-Feb18--Testing.dat'));

%% Load Predictions
foldername = './svm_perf/robotCollision/';
prediction_names = dir('./svm_perf/robotCollision/36d-120k-NewIIWASetup-Feb18-Predictions.dat'); 
nPredictions = length(prediction_names);
predictions  = zeros(length(y_test),nPredictions);
for i=1:nPredictions
    predictions(:,i) = sign(svmlread(strcat(foldername,prediction_names(i).name)));
end

% Dataset Sizes
% M = [ 67000 130000 270000 540000 ];
M = [length(predictions)];

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

%% FOR MULTIPLE MODELS
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

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   STEP 7: Load Chosen Model     %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load Choosen Model (540k k=3000 SVs)
folder_name   = './svm_perf/robotCollision/';
optimal_model_name = '36d-120k-NewIIWASetup-Feb18-Model.dat';
% optimal_model_name = '2d_model.dat';
cpsp_model_robots = readCPSPmodel(strcat(folder_name,optimal_model_name));
