function [D, D_pos, D_neg, D_btw] = computePairwiseDistances(X, y, display_hist)

maxSamples = 10000;
if length(y) < maxSamples
    X_train = X; y_train = y;
else
    X_train = X(1:maxSamples, :); y_train = y(1:maxSamples);
end
size(X_train)

%%%%% Compute Element-wise pairwise distances %%%%%%
%%% Throughout ALL training points %%%
tic;
D = pdist(X_train, 'euclidean');
toc;
mean_D = mean(D(:))

%%% Within Positive Class training points %%%
tic;
D_pos = pdist(X_train(y_train==1,:), 'euclidean');
toc;
mean_Dpos = mean(D_pos(:));

%%% Within Negative Class training points %%%
tic;
D_neg = pdist(X_train(y_train==-1,:), 'euclidean');
toc;
mean_Dneg = mean(D_neg(:))

%%% Between Class training points %%%
tic;
D_btw = pdist2(X_train(y_train==-1,:), X_train(y_train==1,:), 'euclidean');
toc;
mean_Dbtw = mean(D_btw(:))

% Visualize pairwise distances as Histogram
if (display_hist == 1)
figure('Color',[1 1 1])
hist_distances = 10;


D_v = D(:);
tot_distances  = length(D_v);

subplot(2,2,1)
histfit(D_v(1:hist_distances:end,:))
title('Collision Avoidance Features Pairwise Distances')
xlabel('L_2 Norm')
grid on 
axis tight

subplot(2,2,2)
D_v = D_pos(:);
tot_distances  = length(D_v);
histfit(D_v(1:hist_distances:end,:))
title('Collision Avoidance Features Pairwise Distances (Positive Class)')
xlabel('L_2 Norm')
grid on 
axis tight


subplot(2,2,3)
D_v = D_neg(:);
tot_distances  = length(D_v);
histfit(D_v(1:hist_distances:end,:))
title('Collision Avoidance Features Pairwise Distances (Negative Class)')
xlabel('L_2 Norm')
grid on 
axis tight

subplot(2,2,4)
D_v = D_btw(:);
tot_distances  = length(D_v);
histfit(D_v(1:hist_distances:end,:))
title('Collision Avoidance Features Pairwise Distances (Between Class)')
xlabel('L_2 Norm')
grid on 
axis tight
end

end