% ELM function was written by U.Yuzgec, 25.10.2023
% ELM_func(data,Neurons_HL,Input_Features,AF,type)
% This code was inspired from the paper given below:
% Guang-Bin Huang, Qin-Yu Zhu, Chee-Kheong Siew, Extreme learning machine: 
% Theory and applications, Neurocomputing, Volume 70, Issues 1–3, 2006, Pages 489-501

function [ModelOutputs_train,ModelOutputs_test,time] = ELM_func(data,Neurons_HL,Input_Features,AF,type)
 
xtrain = data.TrainInputs;
xtest = data.TestInputs;
ytrain = data.TrainTargets;
ytest = data.TestTargets;

Ns_train = length(ytrain);
Ns_test = length(ytest);
[xtrain,mux,sigmax] = zscore(xtrain);       % z-score normalization
[ytrain,muy,sigmay] = zscore(ytrain);
 xtest=(xtest-mux)./sigmax;                 % test data is normalized by z-score
 ytest=(ytest-muy)./sigmay; 

time=[];
ModelOutputs_train = zeros(Ns_train,1);
ModelOutputs_test = zeros(Ns_test,1);

% start training ELM model...
    tstart = tic; 
    Inputweights=rand(Neurons_HL,Input_Features)*2-1; % randomly generated input weights
    Bias_HL=rand(Neurons_HL,1);Biasmatrix=Bias_HL(:,ones(1,Ns_train)); % randomly generated biases

    Prod=xtrain*Inputweights';
    H=Prod+Biasmatrix';   % output of the hidden layer

    if strcmp(AF,'tanh')
        Hout=tanh(H);
    elseif strcmp(AF,'sig')
        Hout=1./(1+exp(-H));
    elseif strcmp(AF,'sin')
        Hout=sin(H);
    elseif strcmp(AF,'cos')
        Hout=cos(H);
    elseif strcmp(AF,'RBF')
        Hout=radbas(H);
    elseif strcmp(AF,'tf')
        Hout=tribas(H);
    elseif strcmp(AF,'logsig')
        Hout=logsig(H);
    end

    if strcmp(type,'MP')
        Hinv=pinv(Hout);
    elseif strcmp(type,'RCOD')
        Hinv=RCOD(Hout);
    elseif strcmp(type,'OT')
        lambda=10000;
        Hinv=ORT(Hout,lambda);
    end
	
	time = toc(tstart); % calculating the training time of model
    Outputweights=(Hinv)*ytrain;
    ModelOutputs_train = Hout*Outputweights;  % ELM outputs predicted on the training dataset  
   
% testing the ELM model
    Prod=xtest*Inputweights';
    H=Prod+Bias_HL(:,ones(1,Ns_test))';  

    if strcmp(AF,'tanh')            % chosing activation function
        Hout=tanh(H);
    elseif strcmp(AF,'sig')
        Hout=1./(1+exp(-H));
    elseif strcmp(AF,'sin')
        Hout=sin(H);
    elseif strcmp(AF,'cos')
        Hout=cos(H);
    elseif strcmp(AF,'RBF')
        Hout=radbas(H);
    elseif strcmp(AF,'tf')
        Hout=tribas(H);
    elseif strcmp(AF,'logsig')
        Hout=logsig(H);    
    end

    ModelOutputs_test = Hout*Outputweights;

% z-score denormalization
ModelOutputs_train = ModelOutputs_train*sigmay+muy;
ModelOutputs_test = ModelOutputs_test*sigmay+muy;