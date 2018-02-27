%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Generate 2D Concentric Region Dataset (Ideal Example of collision Regions)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all;
close all;

dim_samples = 2;
num_classes = 2;

offsets      = [0 0 ; -5 3; 3 5; -2 8];
nb_samples  = [200; 500; 200; 500];
X = []; labels = [];


for i=1:4       
    num_samples = nb_samples(i);    
    [X_tmp,labels_tmp]  = ml_circles_data(num_samples,dim_samples,num_classes);
    labels_bin          = ml_2binary(labels_tmp)';    
    
    X_tmp = bsxfun(@plus, X_tmp, offsets(i,:));
    
    X = [X; X_tmp];
    labels = [labels; labels_bin];

end

% Randomize indices of dataset
rand_idx = randperm(length(X));
X = X(rand_idx,:);
labels = labels(rand_idx);

% Visualize data
plot_options            = [];
plot_options.is_eig     = false;
plot_options.labels     = labels;
plot_options.title      = 'Ideal 2D Collision Regions';

if exist('h1','var') && isvalid(h1), delete(h1);end
h1 = ml_plot_data(X,plot_options);
legend('Class -1', 'Class 1')
axis equal

%% Find range for rbf kernel
pairwise_distances = zeros(1,length(X));
for j = 1:length(X)
    pairwise_distances(j) = norm(X(1,:) - X(j,:));
end
[sorted_distances, sorted_index] = sort(pairwise_distances);
figure('Color',[1 1 1])
nbIDx = length(X)*0.5;
plot(1:nbIDx,sorted_distances(1:nbIDx),'--b');
title('Pairwise Distances')
xlabel('idx'); ylabel('L_2 Norm')
grid on 
axis tight

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                     C - SUPPORT VECTOR MACHINE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Test C-SVM on Data
clear options
options.svm_type    = 0;    % 0: C-SVM, 1: nu-SVM
options.C           = 100;  % C misclasification penalty [1-1000]
options.sigma       = 1.3;  % radial basis function: exp(-gamma*|u-v|^2), gamma = 1/(2*sigma^2)

% Train SVM Classifier (30k pts 15/18s)
tic;
[predict_labels, model] = svm_classifier(X, labels, options, []);
toc;

% Plot SVM Boundary from ML_toolbox
ml_plot_svm_boundary(X, labels, model, options, 'draw');

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Do grid-search to find 'optimal' hyper-parameters for nu-SVM with RBF
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set options for SVM Grid Search and Execute
clear options
options.svm_type   = 1;           % SVM Type (0:C-SVM, 1:nu-SVM)
options.limits_nu  = [0.01, 0.5]; % Limits of penalty nu
options.limits_w   = [0.5, 2.5];    % Limits of kernel width \sigma
options.steps      = 10;          % Step of parameter grid 
options.K          = 10;          % K-fold CV parameter

% Do Grid Search
clear ctest ctrain
tic;
[ ctest, ctrain , cranges ] = ml_grid_search_class( X, labels, options );
toc;

%% Extract parameter ranges
nu_range = cranges(1,:);
w_range  = cranges(2,:);

% Get CV statistics 
stats = ml_get_cv_grid_states(ctest,ctrain);

% Visualize Grid-Search Heatmap
cv_plot_options              = [];
cv_plot_options.title        = strcat('\nu-SVM :: ', num2str(options.K),'-fold CV with RBF');
cv_plot_options.param_names  = {'\nu', '\sigma'};
cv_plot_options.param_ranges = [nu_range ; w_range];

if exist('hcv','var') && isvalid(hcv), delete(hcv);end
hcv = ml_plot_cv_grid_states(stats,cv_plot_options);

% Find 'optimal hyper-parameters'
[max_acc,ind]   = max(stats.test.acc.mean(:));
[nu_max, w_max] = ind2sub(size(stats.test.acc.mean),ind);
nu_opt = nu_range(nu_max);
w_opt  = w_range(w_max);

%% Plot Decision Boundary with 'Optimal' Hyper-parameters from nu-SVM CV
clear options

% Chosen from heatmap (Gives less support vectors)
nu_opt = 0.1733;
w_opt  = 0.7222;

options.svm_type = 1;    
options.nu       = nu_opt;
options.sigma    = w_opt; 

% Train SVM Classifier
[predict_labels, model] = svm_classifier(X, labels, options, []);

% Plot SVM decision boundary
ml_plot_svm_boundary( X, labels, model, options, 'surf');
