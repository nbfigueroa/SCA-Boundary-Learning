function [cpsp_model] =  readCPSPmodel(model_name)

% Read SVM-light Model
fID = fopen(model_name);
tline = fgetl(fID);
tlines = cell(0,1);
while ischar(tline)
    tlines{end+1,1} = tline;
    tline = fgetl(fID);
end
fclose(fID);

tmp   = strsplit(tlines{4});
gamma = str2double(tmp{1});

tmp   = strsplit(tlines{8});
D     = str2num(tmp{1});


tmp     = strsplit(tlines{9});
Nbvs_tr = str2num(tmp{1});

tmp   = strsplit(tlines{10});
Nbvs  = str2num(tmp{1}) - 1;

tmp   = strsplit(tlines{11});
bias  = str2double(tmp{1});

alphay = zeros(Nbvs, 1);
BVs    = zeros(Nbvs, D);

clc;
fprintf('Reading a CPSP model with %d basis vectors, of which %d are training data.\n',Nbvs, Nbvs_tr);

for i=1:Nbvs
    tmp   = strsplit(tlines{11+i});
    alphay(i,1) = str2num(tmp{1});    
    
    for j=2:D+1
        featVal = strsplit(tmp{j},':')  ; 
        if strcmp(featVal{1}, '#')
            break;
        else
            featID    = str2double(featVal{1});
            value     = featVal{2};
            featValue = str2double(value);
            BVs(i,featID) = featValue;
        end
    end
end

cpsp_model.gamma  = gamma;
cpsp_model.D      = D;
cpsp_model.Nbvs   = Nbvs;
cpsp_model.bias   = bias;
cpsp_model.alphay = alphay;
cpsp_model.BVs    = BVs;

fprintf('Model Parameters: Gaussian Kernel with gamma=%3.3f and bias=%3.3f.\n', gamma, bias);
end