clc
clear 
close all

load BreastCancer.mat

N = numel(T);
PR = randperm(N);

T = T(PR);
X = X(PR,:);

pTrain = 0.7;

Ix1 = find(T==0)';
Ix2 = find(T==1)';

TrainInd1 = Ix1(1:round(pTrain*numel(Ix1)));
TrainInd2 = Ix2(1:round(pTrain*numel(Ix2)));
TrainInd = [TrainInd1 TrainInd2];


TestInd1 = Ix1(round(pTrain*numel(Ix1))+1:end);
TestInd2 = Ix2(round(pTrain*numel(Ix2))+1:end);
TestInd = [TestInd1 TestInd2];

SVMStruct = fitcsvm(X(TrainInd,:),T(TrainInd),'Standardize',true,'KernelFunction','RBF');
Y_tr = predict(SVMStruct,X(TrainInd,:));
Y_ts = predict(SVMStruct,X(TestInd,:));
% 
% RAWSVM = templateSVM('KernelFunction','rbf');  % *** MULTI-CLASS *** %
% SVMStruct2 = fitcecoc(X(TrainInd,:),T(TrainInd),'Learners',RAWSVM);
% Y_tr2 = predict(SVMStruct2,X(TrainInd,:));
% Y_ts2 = predict(SVMStruct2,X(TestInd,:));

[Err_tr, CM_tr] = confusion(T(TrainInd)',Y_tr')
[Err_ts, CM_ts] = confusion(T(TestInd)',Y_ts')
%Accuracy
Acc_tr = (1-Err_tr)*100
Acc_ts = (1-Err_ts)*100
%sensitivity
Sn_tr = CM_tr(2,2)/sum(CM_tr(2,:))*100
Sn_ts = CM_ts(2,2)/sum(CM_ts(2,:))*100
%specificity
Sp_tr = CM_tr(1,1)/sum(CM_tr(1,:))*100
Sp_ts = CM_ts(1,1)/sum(CM_ts(1,:))*100


