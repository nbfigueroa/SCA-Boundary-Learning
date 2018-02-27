function [X, y, X_test, y_test] = LoadCollisionDatasets(dataset_name, sample_col)

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Prepare Collision Region Dataset for Learning %%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

colided1_fname = strcat(dataset_name, '/position1_colided01.txt');
colided2_fname = strcat(dataset_name,'/position2_colided01.txt');
tmp_r1 = importfile_debug(colided1_fname);
tmp_r2 = importfile_debug(colided2_fname);

fprintf('Collided Points: %d \n', size(tmp_r1,1));
% Sub-sample Dataset for Training Set
% sample_col = 2;
R1 = tmp_r1(1:sample_col:end,:);
R2 = tmp_r2(1:sample_col:end,:);

% Sub-sample Dataset for Testing Set
R1_test = tmp_r1(2:sample_col:end,:);
R2_test = tmp_r2(2:sample_col:end,:);

% Use 3D position data x(q)
Collision_regions = [R1 R2];
labels_Collision    = ones(length(Collision_regions),1)*-1; 

Collision_regions_test = [R1_test R2_test];
labels_Collision_test  = ones(length(Collision_regions_test),1)*-1; 

fprintf('Train: %d  Test:%d \n', length(labels_Collision), length(labels_Collision_test) );

%%%%%%%% Loading Neighbor Regions %%%%%%
tmp_r1 = importfile_debug(strcat(dataset_name, '/position1_neighbour01.txt'));
tmp_r2 = importfile_debug(strcat(dataset_name,'/position2_neighbour01.txt'));

fprintf('Neighbor Points: %d \n', size(tmp_r1,1));

% Sub-sample Dataset
R1 = tmp_r1(1:sample_col:end,:);
R2 = tmp_r2(1:sample_col:end,:);

R1_test = tmp_r1(2:sample_col:end,:);
R2_test = tmp_r2(2:sample_col:end,:);

% Use 3D position data x(q)
NoCollision_regions = [R1 R2];
labels_Nocollision  = ones(length(NoCollision_regions),1); 

NoCollision_regions_test = [R1_test R2_test];
labels_Nocollision_test  = ones(length(NoCollision_regions_test),1); 

fprintf('Train: %d  Test: %d\n', length(labels_Nocollision), length(labels_Nocollision_test) );

%%%%%% Collision + Non Collision Regions for Training Set %%%%%%
X = [Collision_regions; NoCollision_regions];
y = [labels_Collision;  labels_Nocollision];

% Randomize indices
randidx = randperm (length(y));
X = X(randidx,:); y = y(randidx,1);

%%%%%% Collision + Non Collision Regions for Testing Set %%%%%%
X_test = [Collision_regions_test; NoCollision_regions_test];
y_test = [labels_Collision_test;  labels_Nocollision_test];

% Randomize indices
randidx = randperm (length(y_test));
X_test = X_test(randidx,:); y_test = y_test(randidx,1);

end