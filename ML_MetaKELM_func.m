% Multilayer Meta Kernel ELM function was written by U.Yuzgec, 06.11.2023
% ML_MetaKELM_func(data,Neurons_HL,AF,type,L,M,Kernel_type,Kernel_parameter,C_i)

function [ModelOutputs_train,ModelOutputs_test,time] = ML_MetaKELM_func(data,Neurons_HL,AF,type,L,M,Kernel_type,Kernel_parameter,C_i)
 
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

DS = round(Ns_train/M);			% Data samples
C=ones(1,L)*C_i;

time=[];
ModelOutputs_train = zeros(Ns_train,1);
ModelOutputs_test = zeros(Ns_test,1);

% start training Multilayer Meta Kernel ELM model...
    tstart = tic; 
   	Xtemp = xtrain;
	for m=1:L-1 %do it for each layer
		for j=1:M,
			if j==1 
				Xm = Xtemp(1:j*DS,:); 
			else 
				Xm = Xtemp((j-1)*DS+1:min(j*DS, max(size(Xtemp))),:); 
			end
			Nm = size(Xm,1); % the number of subset samples : Ns_train/M
			n = size(Xm,2);  % the number of external inputs
		
			Inputweights(:,:,j) = rand(Neurons_HL,n)*2-1;		% randomly generated input weights
			Bias_HL(:,:,j) = rand(Neurons_HL,1);
% % % 			Biasmatrix(:,:,j) = Bias_HL(:,ones(1,Nm),j);  % randomly generated biases

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
				Outputweights(:,:,j) = (Hinv)*Xm;
			else
				Outputweights(:,:,j) = (Hinv)*Xm;
			end
		end % for j=1:M...
	
		for j=1:M,
			Prod=Xtemp*Inputweights(:,:,j)';
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
			Htrain(:,:,j) = Hout*Outputweights(:,:,j);	
        end
        
        for r=1:n
            gec=[];
            for k=1:M
                gec=[gec Htrain(:,r,k)];
            end
			G(:,:,r)=gec;
            % Training of ML Meta KELM
            Omega_train = kernel_matrix(G(:,:,r),Kernel_type,Kernel_parameter);
            OutputW(:,r) = ((Omega_train+speye(Ns_train)/C(m))\(Xtemp(:,r)));
            Xtemp(:,r) = Omega_train*OutputW(:,r);
        end
    	IW{m} =	Inputweights;
		IB{m} = Bias_HL;
		OW{m} =	Outputweights;
        OutW{m} = OutputW;
		GG{m} = G;
        clear Inputweights
        clear Biasmatrix
        clear Bias_HL
        clear Outputweights
        clear Htrain   
        clear OutputW
	end % for m
	%	------------------------------------------------------------------------------- 
	%% Stage:2 Training of the last layer of Multi Layer Meta KELM
	    for j=1:M,
		if j==1 
			Xm = Xtemp(1:j*DS,:); 
		else 
			Xm = Xtemp((j-1)*DS+1:min(j*DS, max(size(Xtemp))),:); 
		end
		Nm = size(Xm,1); % the number of subset samples : Ns_train/M
		n = size(Xm,2);  % the number of external inputs
		
		Finalweights(:,:,j) = rand(Neurons_HL,size(Xtemp,2))*2-1;		% randomly generated input weights
		FinalBias_HL(:,:,j) = rand(Neurons_HL,1);

		Prod=Xm*Finalweights(:,:,j)';
        H=Prod+FinalBias_HL(:,ones(1,Nm),j)';		% output of the hidden layer
        
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
			FinalOutputweights(:,:,j) = (Hinv)*ytrain(1:DS);
		else
			FinalOutputweights(:,:,j) = (Hinv)*ytrain(((j-1)*DS+1):min(j*DS, max(size(Xtemp))));
		end
	end % for j=1:M...
	
	for j=1:M,
		Prod=Xtemp*Finalweights(:,:,j)';
		H=Prod+FinalBias_HL(:,ones(1,Ns_train),j)';		% output of the hidden layer
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
		HtrainFinal(:,j) = Hout*FinalOutputweights(:,:,j);	
	end
      
	Omega_train = kernel_matrix(HtrainFinal,Kernel_type,Kernel_parameter);
	Outputweights_Final=((Omega_train+speye(Ns_train)/C(end))\(ytrain)); 
	
	time = toc(tstart); % calculating the training time of model

	% Multi Layer Meta Kernel ELM outputs predicted on the training dataset
	ModelOutputs_train = Omega_train*Outputweights_Final;
	     
  % testing the Multi Layer Meta Kernel ELM model
	
	Xtemp = xtest;
	for m=1:L-1 %do it for each layer
		Inputweights =	IW{m};
		Biasmatrix = IB{m};
		Outputweights =	OW{m};
        OutputW = OutW{m};
        G = GG{m};
		for j=1:M,
			if m==1
                Prod=Xtemp*Inputweights(:,:,j)';
            else
                Prod=Xtemp*Inputweights(:,:,j)';    
            end
			H=Prod+Biasmatrix(:,ones(1,Ns_test),j)';		 
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
			Htest(:,:,j) = Hout*Outputweights(:,:,j);	
		end % for j
        
        for r=1:n
            gec=[];
            for k=1:M
                gec=[gec Htest(:,r,k)];
            end
			Omega_test = kernel_matrix(G(:,:,r),Kernel_type,Kernel_parameter,gec);
            Xtemp(:,r) = Omega_test'*OutputW(:,r);
         end

		clear Inputweights
        clear Biasmatrix
        clear Outputweights
	end % for m
	
	for j=1:M,
		Prodfinal = Xtemp*Finalweights(:,:,j)';
		Hfinal = Prodfinal+FinalBias_HL(:,ones(1,Ns_test),j)';		 
		if strcmp(AF,'tanh')
			Hout=tanh(Hfinal);
		elseif strcmp(AF,'sig')
			Hout=1./(1+exp(-Hfinal));
		elseif strcmp(AF,'sin')
			Hout=sin(Hfinal);
		elseif strcmp(AF,'cos')
			Hout=cos(Hfinal);
		elseif strcmp(AF,'RBF')
			Hout=radbas(Hfinal);
		elseif strcmp(AF,'tf')
			Hout=tribas(Hfinal);
		elseif strcmp(AF,'logsig')
			Hout=logsig(Hfinal);
		end
		HtestFinal(:,j) = Hout*FinalOutputweights(:,:,j);	
	end
	
	Omega_test = kernel_matrix(HtrainFinal,Kernel_type,Kernel_parameter,HtestFinal);
	ModelOutputs_test = Omega_test'*Outputweights_Final;
	
% z-score denormalization
ModelOutputs_train = ModelOutputs_train*sigmay+muy;
ModelOutputs_test = ModelOutputs_test*sigmay+muy;