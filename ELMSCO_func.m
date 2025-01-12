% ELM-SCO function was written by U.Yuzgec, 15.12.2023
% ELMSCO_func(data,Neurons_HL,Input_Features,AF,type)

function [ModelOutputs_train,ModelOutputs_test,time] = ELMSCO_func(data,Neurons_HL,Input_Features,AF,type)
 
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
refresh=10; 
ModelOutputs_train = zeros(Ns_train,1);
ModelOutputs_test = zeros(Ns_test,1);

Max_iter=100;
conv_curve=zeros(1,Max_iter);
POO=0; %% Intial counter to count unsuccessful fitness improvements
m=5; %% number of unsuccessful attempts to improve the fitness
alpha=round(Max_iter/3); % number of function evaluations in the First phase
b=2.4;
S=[];

% start training ELMSCO model...
tstart = tic;
Inputweights=rand(Neurons_HL,Input_Features)*2-1; % randomly generated input weights
Bias_HL=rand(Neurons_HL,1); % randomly generated biases
for r=1:Input_Features
    S=[S Inputweights(:,r)']; % ELM weights...
end
S=[S Bias_HL'];
dim=size(S,2);
ub=1*ones(1,dim);
lb=-1*ones(1,dim);
Range=ub-lb;

[~,~,BF] = ELM(data,Inputweights,Bias_HL,AF,type);

for t=1:Max_iter
   w(t) =exp(-(b*t/Max_iter)^b); %% Equation (3) in the paper 
     if t>alpha
		  if sum(P)==0       %% Counting the number of unsuccessful fitness improvements
			  POO=1+POO;      %% Counter to count unsuccessful fitness improvements
		  end
     end
		K=rand;
    for j = 1:dim
		EE= w(t)*K*Range(j);
    if t<alpha 
        if rand<0.5
			x(j) = S(j)+(w(t)*abs(S(j)));
        else   %% Equation (2) in the paper 
			x(j) = S(j)-(w(t)*abs(S(j)));
        end 
            
    else
        if POO==m
            POO=0;      %% Reset counter
        if rand<0.5
            x(j) = S(j)+rand*Range(j); 
        else                           %% Equation (5) in the paper
            x(j) = S(j)-rand*Range(j);
        end 
        else
          if rand<0.5
             x(j)=S(j)+EE; 
          else                                %% Equation (4) in the paper
             x(j)=S(j)-EE;
          end   
        end
    
    end
    
    %% Check if a dimension of the candidate solution goes out of boundaries
       if x(j)>ub(j)
           x(j)=S(j);
       end              %% Equation (6) in the paper
        if x(j)<lb(j)
           x(j)=S(j);
        end
    end
    
 %% Evaluate the fitness of the newly generated candidate solution
 
    for r=1:size(Inputweights,2)
       Inputweights_x(:,r) = x((r-1)*Neurons_HL+1:Neurons_HL*r)';
    end
    Bias_HL_x = x((r)*Neurons_HL+1:Neurons_HL*(r+1))';
   	
	[~,~,F] = ELM(data,Inputweights_x,Bias_HL_x,AF,type);
          
    if F < BF
        BF = F;
		S = x;
        P = 1;
    else
        P = 0;
    end
    
    conv_curve(1,t) = BF;
    % Show Iteration Information
	if (rem(t,refresh)==0)
		disp(['Iteration ' num2str(t) ': Best Cost = ' num2str(BF)]);
	end
end     	
	
time = toc(tstart); % calculating the training time of model	
%% Evaluate the fitness of the final candidate solution
    for r=1:size(Inputweights,2)
       Inputweights(:,r) = S((r-1)*Neurons_HL+1:Neurons_HL*r)';
    end
    Bias_HL = S((r)*Neurons_HL+1:Neurons_HL*(r+1))';
	[ModelOutputs_train,ModelOutputs_test,F] = ELM(data,Inputweights_x,Bias_HL_x,AF,type);

% z-score denormalization
ModelOutputs_train = ModelOutputs_train*sigmay+muy;
ModelOutputs_test = ModelOutputs_test*sigmay+muy;
end

%-------------------------------------------------------------------------------------------------
function [ModelOutputs_train,ModelOutputs_test,BF] = ELM(data,Inputweights,Bias_HL,AF,type)
 
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

ModelOutputs_train = zeros(Ns_train,1);
ModelOutputs_test = zeros(Ns_test,1);

% start training ELM model...

    %%%%%%%%%%%%%%% Inputweights=rand(Neurons_HL,Input_Features)*2-1; % randomly generated input weights
    %%%%%%%%%%%%%%% Bias_HL=rand(Neurons_HL,1);
	Biasmatrix=Bias_HL(:,ones(1,Ns_train)); % randomly generated biases

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
% calculating the mse...
BF = errperf(ytrain,ModelOutputs_train,'mse');
end

