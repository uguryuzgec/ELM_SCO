%       Near Real-time Machine Learning Framework in Distribution         %
%      Networks with Low-Carbon Technologies Using Smart Meter Data       %
%  Emrah Dokur, Nuh Erdogan,Ibrahim Sengor, Ugur Yuzgec & Barry P. Hayes  %  
%             This article was submitted to Applied Energy                %
%             This code was written by E. Dokur & U.Yuzgec                %
%-------------------------------------------------------------------------%

clc
clear
close all
warning off

%% Choose ELM models
% 1.  ELM model
% 2.  Meta ELM model
% 3.  Multi Layer Meta Kernel ELM model
% 4.  ELM-SCO model (Proposed Model)

Options = {'ELM model','Meta ELM model',...
    'Multi Layer Meta Kernel ELM model','ELM-SCO model'};

[Selection, Ok] = listdlg('PromptString', 'Select ELM model:', ...
                          'SelectionMode', 'single','ListSize',[200,250], ...
                          'ListString', Options);
pause(0.01);

if Ok==0
    return;
end

runs = 1;

data=LoadData_ISGT(); 

ytrain = data.TrainTargets;
ytest = data.TestTargets;

%% ELM parameters...
Neurons_HL =35;
Input_Features = size(data.TrainInputs,2);
AF = 'tanh';
type = 'OT';
%% KELM parameters...
Kernel_type = 'RBF_kernel';
% 'RBF_kernel' for Gauss Kernel
% 'lin_kernel' for Linear Kernel
% 'poly_kernel' for Polynomial Kernel
% 'wav_kernel' for Wavelet Kernel
Kernel_parameter = [50 2 pi*0.533]; 
C = 50;  % Regularization_coefficient=1;

%% Multilayer ELM parameters...
% the paramemeters are the same as ELM model without Input_Features...
L = 3; % number of layer in the network

%% Meta ELM parameters...
% the paramemeters are the same as ELM model without Neurons_HL...
M = 5; % Number of Meta ELM groups
Neurons_HL_Meta = 10; % Neurons in the hidden layer...

   
%% Run selected model...
    switch Selection
        case 1, 
            for i=1:runs
                disp('runs: '+string(num2str(i))); 
                [ModelOutputs_train(:,i),ModelOutputs_test(:,i),time(i)] = ...
                              ELM_func(data,Neurons_HL,Input_Features,AF,type);
            end  
            str=string('ELM Model');
        case 2, 
            for i=1:runs
                disp('runs: '+string(num2str(i))); 
                [ModelOutputs_train(:,i),ModelOutputs_test(:,i),time(i)] = ...
                              MetaELM_func(data,Neurons_HL_Meta,AF,type,M);
            end   
            str=string('Meta ELM Model');
        case 3, 
            for i=1:runs
                disp('runs: '+string(num2str(i))); 
                [ModelOutputs_train(:,i),ModelOutputs_test(:,i),time(i)] = ...
                              ML_MetaKELM_func(data,Neurons_HL_Meta,AF,type,L,M,Kernel_type,Kernel_parameter,C);
            end % for... 
            str=string('Multi Layer Meta Kernel ELM model');
        case 4, 
            for i=1:runs
                disp('runs: '+string(num2str(i))); 
                [ModelOutputs_train(:,i),ModelOutputs_test(:,i),time(i)] = ...
                              ELMSCO_func(data,Neurons_HL,Input_Features,AF,type);
            end % for...
            str=string('ELM SCO model');
    end %  switch Selection...
  
    for i=1:runs,     
        mse_train(i)  = errperf(ytrain,ModelOutputs_train(:,i),'mse');
        rmse_train(i) = errperf(ytrain,ModelOutputs_train(:,i),'rmse');
        mae_train(i)  = errperf(ytrain,ModelOutputs_train(:,i),'mae');
        r2_train(i) = calculateR2(ytrain,ModelOutputs_train(:,i)); 
    
        mse_test(i)  = errperf(ytest,ModelOutputs_test(:,i),'mse');
        rmse_test(i) = errperf(ytest,ModelOutputs_test(:,i),'rmse');
        mae_test(i)  = errperf(ytest,ModelOutputs_test(:,i),'mae');
        r2_test(i) = calculateR2(ytest,ModelOutputs_test(:,i)); 
    end

[~,index1]=min(mse_train(:)); % best train performance of the selected model...
[~,index2]=min(mse_test(:));  % best test performance of the selected model...
  
figure(1)
plot(ytrain,'-.k','LineWidth',2);
set(gcf,'color','w', 'units','normalized','outerposition',[0 0 1 1]);
hold on;
plot(ModelOutputs_train(:,index1),'b','LineWidth',1);
legend('Target',str);
title(string('Training Result of ')+str);
xlabel('Time (hours)');
ylabel('Wind Power (MW)');
axis tight

figure(2)
plot(ytest,'-.k','LineWidth',2);
set(gcf,'color','w', 'units','normalized','outerposition',[0 0 1 1]);
hold on;
plot(ModelOutputs_test(:,index2),'b','LineWidth',1);
legend('Target',str);
title(string('Test Result of ')+str);
xlabel('Time (hours)');
ylabel('Wind Power (MW)');
axis tight
 
PlotResults(ytrain,ModelOutputs_train(:,index1),string('Training Result of ')+str);
set(gcf,'color','w', 'units','normalized','outerposition',[0 0 1 1]);
PlotResults(ytest,ModelOutputs_test(:,index2),string('Test Result of ')+str);
set(gcf,'color','w', 'units','normalized','outerposition',[0 0 1 1]);

figure
subplot(121)
scatter(ytrain,ModelOutputs_train(:,index1));
set(gcf,'color','w', 'units','normalized','outerposition',[0 0 0.5 1]);
xlabel('Observed Data');
ylabel('Estimated Data');
title(string('Training Performance of ')+str);
box on
subplot(122)
scatter(ytest,ModelOutputs_test(:,index2));
set(gcf,'color','w', 'units','normalized','outerposition',[0 0 0.5 1]);
xlabel('Observed Data');
ylabel('Estimated Data');
title(string('Test Performance of ')+str);
box on

Train_Results = table(categorical({'MSE';'RMSE';'MAE';'R2';'Run Time'}),...
[min(mse_train);min(rmse_train);min(mae_train);max(r2_train);min(time)],...
[max(mse_train);max(rmse_train);max(mae_train);min(r2_train);max(time)],...
[median(mse_train);median(rmse_train);median(mae_train);median(r2_train);median(time)],...
[mean(mse_train);mean(rmse_train);mean(mae_train);mean(r2_train);mean(time)],...
[std(mse_train);std(rmse_train);std(mae_train);std(r2_train);std(time)],...
'VariableNames',{'Metrics' 'Best' 'Worst' 'Median' 'Mean' 'Std'});

Test_Results = table(categorical({'MSE';'RMSE';'MAE';'R2'}),...
[min(mse_test);min(rmse_test);min(mae_test);max(r2_test)],...
[max(mse_test);max(rmse_test);max(mae_test);min(r2_test)],...
[median(mse_test);median(rmse_test);median(mae_test);median(r2_test)],...
[mean(mse_test);mean(rmse_test);mean(mae_test);mean(r2_test)],...
[std(mse_test);std(rmse_test);std(mae_test);std(r2_test)],...
'VariableNames',{'Metrics' 'Best' 'Worst' 'Median' 'Mean' 'Std'});

disp(string('Train Results of ')+str);
Train_Results 

disp(string('Test Results of ')+str);
Test_Results
