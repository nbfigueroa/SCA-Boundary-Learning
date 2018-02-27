%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%                LEARN COLLISION/NON-COLLISION REGION DATA               %                                                                       
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%           STEP 1: LOAD DATASET                %
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all; clc;
load_option = 1; % 0: Loads joint positions from text files, randomly samples 
                 % 'sample_col' of the dataset and generates training and testing sets
                 % by splitting the dataset in 1/2 for EACH Class: collided (y=-1) / non-collided (y=+1)   
                 % 1: Loads a mat file with the dataset as above  

sample_col = 1;  % Variable to set the sub-sampling size of dataset, should be in integers
                 % i.e. 1/sample_col will be extracted   
                 
switch load_option
    case 0
        % For Joint Collision Testing (Arms closer)
        % dataset_name = '../function_learning/collisionDatasets/data/fOR_TEST';
        
        % KUKA Innovation Award Setup - 20 deg resolution
        % dataset_name = '../function_learning/collisionDatasets/data/New_innovation_award';
        
        % New LASA lab Dual-Arm IIWA setup (Feb 2018) - 40 deg resolution
%         dataset_name = '../function_learning/collisionDatasets/data/New_IIWA_Setup_Feb18';                
        [X, y, X_test, y_test] = LoadCollisionDatasets(dataset_name, sample_col);

    case 1
        % KUKA Innovation Award Setup - with sample_col = 2
        % dataset_name = '../function_learning/collisionDatasets/data_mat/Innovation_Award_Dataset.mat';
        
        % New LASA lab Dual-Arm IIWA setup (Feb 2018) - with sample_col = 1 - 40 deg resolution
        dataset_name = '../function_learning/collisionDatasets/data_mat/New_IIWA_Setup_Feb18_Dataset.mat';        
        load(dataset_name)
end
 
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  STEP 2: Partition Dataset into Train+Validation/Test  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tt_ratio = 0.02; % Learn model on tt_ratio*10% of the dataset
[ X_train, y_train, X_valid, y_valid ] = split_data(X', y', tt_ratio );

X_train = X_train'; y_train = y_train';
X_valid  = X_valid';  y_valid  = y_valid';

%% Use all data for training
X_train = [X; X_test]; y_train = [y;y_test];

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% STEP 3: Find suitable range for rbf kernel   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
display_hist = 1;
[D, D_pos, D_neg, D_btw] = computePairwiseDistances(X_train, y_train, display_hist);

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Validate Range with "Optimal" RBF width size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Supposed Analytical Equation to find an optimal Sigma
% Optimum can only be found if B_bar > W_bar
weight_btw_sep = 0.95;
[optSigma B_bar W_bar] = sigmaSelection(X_train,y_train','Analytical', weight_btw_sep)

% Not really good estimate! But can help with defining the ranges

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%   Grid-search on CV to find 'optimal' hyper-parameters for C-SVM with RBF %
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Set options for SVM Grid Search and Execute
clear options
options.svm_type   = 0;             % SVM Type (0:C-SVM, 1:nu-SVM)
options.limits_C   = [10^-1, 10^4]; % Limits of penalty C
options.limits_w   = [0.05, 2];     % Limits of kernel width \sigma
options.steps      = 10;            % Step of parameter grid 
options.K          = 5;             % K-fold CV parameter
options.log_grid   = 1;             % Log-Spaced grid of Parameter Ranges

% Do Cross-Validarion (K = 1 is pure grid search, K = N is N-CV)
tic;
[ ctest , ctrain , cranges ] = ml_grid_search_class( X_train, y_train, options );
toc;

%% Get CV statistics

% Extract parameter ranges
range_C  = cranges(1,:);
range_w  = cranges(2,:);

% Extract parameter ranges
stats = ml_get_cv_grid_states(ctest,ctrain);

% Visualize Grid-Search Heatmap
cv_plot_options              = [];
cv_plot_options.title        = strcat('36-D, 24k (NEW KUKA IIWA SETUP) --Joint Positions f(q)-- C-SVM :: Grid Search with RBF');
cv_plot_options.param_names  = {'C', '\sigma'};
cv_plot_options.param_ranges = [range_C ; range_w];
cv_plot_options.log_grid     = 1; 

if exist('hcv','var') && isvalid(hcv), delete(hcv);end
hcv = ml_plot_cv_grid_states(stats,cv_plot_options);

% Find 'optimal hyper-parameters'
[max_acc,ind] = max(stats.test.acc.mean(:));
[C_max, w_max] = ind2sub(size(stats.train.acc.mean),ind);
C_opt = range_C(C_max)
w_opt = range_w(w_max)


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Learn Optimal C - SUPPORT VECTOR MACHINE  %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test C-SVM on Data (Assuming you ran CV first)
clear options
% Optimal Values from CV on Xk dataset
options.svm_type    = 0;    % 0: C-SVM, 1: nu-SVM
options.C           = 16.6810; % Misclassification Penalty
options.sigma       = 0.171;  % radial basis function: exp(-gamma*|u-v|^2), gamma = 1/(2*sigma^2)

% Train SVM Classifier (12k+3D pts = 8s,....)
tic;
[y_est, model] = svm_classifier(X_train, y_train, options, []);
toc;

%% OR Load Optimal Model
load('36D-Pos-12k-Optimal-Model.mat')

%% Model Stats
totSV = model.totalSV;
ratioSV = totSV/length(y_train);
posSV = model.nSV(1)/totSV;
negSV = model.nSV(2)/totSV;
boundSV = sum(abs(model.sv_coef) == options.C)/totSV;

fprintf('*SVM Model Statistic*\n Total SVs: %d, SV/M: %1.4f \n +1 SV : %1.4f, -1 SVs: %1.4f, Bounded SVs: %1.4f \n', ...
    totSV, ratioSV, posSV, negSV,  boundSV);

[test_stats] = class_performance(y_train,y_est);
fprintf('*Classifier Performance on Train set (%d points)* \n Acc: %1.5f, F-1: %1.5f, FPR: %1.5f, TPR: %1.5f \n', ...
    length(y_train), test_stats.ACC, test_stats.F1, test_stats.FPR, test_stats.TPR)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%     Evaluate SVM performance on Testing Set   %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extract a random testing points 
maxSamples = 100000;
if length(y_test) < maxSamples
    numSamples = length(y_test);
else
    numSamples = maxSamples;
end

X_test_ = X_test(1:numSamples,:);
y_test_ = y_test(1:numSamples,:);

% Test Learnt Model
[y_est_] = svm_classifier(X_test_, y_test_, [], model);

% Compute Classifier Test Stats
[test_stats] = class_performance(y_test_,y_est_); 
fprintf('*Classifier Performance on Test Set (%d points)* \n Acc: %1.5f, F-1: %1.5f, FPR: %1.5f, TPR: %1.5f \n', ...
    length(y_test_), test_stats.ACC, test_stats.F1, test_stats.FPR, test_stats.TPR)

