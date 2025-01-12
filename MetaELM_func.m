% Meta ELM function was written by U.Yuzgec, 25.10.2023
% MetaELM_func(data,Neurons_HL,AF,type,M)
 
function [ModelOutputs_train,ModelOutputs_test,time] = MetaELM_func(data,Neurons_HL,AF,type,M)
 
xtrain = data.TrainInputs;
xtest = data.TestInputs;
ytrain = data.TrainTargets;
ytest = data.TestTargets;

Ns_train = length(ytrain);
Ns_test = length(ytest);
[xtrain,mux,sigmax] = zscore(xtrain);       % z-score normalization
[ytrain,muy,sigmay] = zscore(ytrain);
xtest=(xtest-mux)./sigmax;       % test data is normalized by z-score
ytest=(ytest-muy)./sigmay; 

DS = round(Ns_train/M);			% Data samples

time=[];
ModelOutputs_train = zeros(Ns_train,1);
ModelOutputs_test = zeros(Ns_test,1);

% start training Meta ELM model...
    tstart = tic; 
    for j=1:M,
		if j==1 
			Xm = xtrain(1:j*DS,:); 
		else 
			Xm = xtrain((j-1)*DS+1:min(j*DS, max(size(xtrain))),:); 
		end
		Nm = size(Xm,1); % the number of subset samples : Ns_train/M
		n = size(Xm,2);  % the number of external inputs
		
		Inputweights(:,:,j) = rand(Neurons_HL,n)*2-1;		% randomly generated input weights
		Bias_HL(:,:,j) = rand(Neurons_HL,1);
% % % 		Biasmatrix(:,:,j) = Bias_HL(:,ones(1,Nm),j);  % randomly generated biases

		Prod=Xm*Inputweights(:,:,j)';
		H=Prod+Bias_HL(:,ones(1,Nm),j)';		% output of the hidden layer

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

		if j==1
			Outputweights(:,:,j) = (Hinv)*ytrain(1:DS);
		else
			Outputweights(:,:,j) = (Hinv)*ytrain(((j-1)*DS+1):min(j*DS, max(size(xtrain))));
		end
	end % for j=1:M...
	
	for j=1:M,
		Prod=xtrain*Inputweights(:,:,j)';
		H=Prod+Bias_HL(:,ones(1,Ns_train),j)';		% output of the hidden layer
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
		Htrain(:,j) = Hout*Outputweights(:,:,j);	
	end
    
	% Training of Meta ELM
    if strcmp(type,'MP')
		Hminv=pinv(Htrain);
	elseif strcmp(type,'RCOD')
		Hminv=RCOD(Htrain);
	elseif strcmp(type,'OT')
		lambda=10000;
		Hminv=ORT(Htrain,lambda);
	end
	
	time = toc(tstart); % calculating the training time of model
	
	Outputweights_Final = (Hminv)*ytrain;   
    ModelOutputs_train = Htrain*Outputweights_Final;  % Meta ELM outputs predicted on the training dataset
 % testing the Meta ELM model
	for j=1:M,
		Prod=xtest*Inputweights(:,:,j)';
		H=Prod+Bias_HL(:,ones(1,Ns_test),j)';		 
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
		Htest(:,j) = Hout*Outputweights(:,:,j);	
	end

    ModelOutputs_test = Htest*Outputweights_Final;
% z-score denormalization
ModelOutputs_train = ModelOutputs_train*sigmay+muy;
ModelOutputs_test = ModelOutputs_test*sigmay+muy;    