clc
clear 
close all

load BreastCancer

x = X';
t = T';

trainFcn = 'trainlm';  

% Create a Pattern Recognition Network
hiddenLayerSize = 20;
net = patternnet(hiddenLayerSize);

% Choose Input and Output Pre/Post-Processing Functions
% For a list of all processing functions type: help nnprocess
net.input.processFcns = {'removeconstantrows','mapminmax'};
net.output.processFcns = {'removeconstantrows','mapminmax'};

% Setup Division of Data for Training, Validation, Testing
% For a list of all data division functions type: help nndivide
net.divideFcn = 'dividerand';  % Divide data randomly
net.divideMode = 'sample';  % Divide up every sample
net.divideParam.trainRatio = 60/100;
net.divideParam.valRatio = 10/100;
net.divideParam.testRatio = 30/100;

% Choose a Performance Function
% For a list of all performance functions type: help nnperformance
net.performFcn = 'mse';

% Choose Plot Functions
% For a list of all plot functions type: help nnplot
net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
    'plotconfusion', 'plotroc'};

net.layers{end}.transferFcn = 'purelin'; 
net.trainParam.max_fail = 10;

% Train the Network
[net,tr] = train(net,x,t);

% Test the Network
y = net(x);

[Err_tr, CM_tr] = confusion(t([tr.trainInd, tr.valInd]),y([tr.trainInd, tr.valInd]))
[Err_ts, CM_ts] = confusion(t(tr.testInd),y(tr.testInd)); 
%Accuracy
Acc_tr = (1-Err_tr)*100
Acc_ts = (1-Err_ts)*100
%sensitivity
Sn_tr = CM_tr(2,2)/sum(CM_tr(2,:))*100
Sn_ts = CM_ts(2,2)/sum(CM_ts(2,:))*100
%specificity
Sp_tr = CM_tr(1,1)/sum(CM_tr(1,:))*100
Sp_ts = CM_ts(1,1)/sum(CM_ts(1,:))*100
