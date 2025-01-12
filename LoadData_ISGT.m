function data_output=LoadData_ISGT(Start,End)

if nargin < 1 || isempty(Start) || Start<=0 || Start>75285
   Start = 10000; % default value
end

if nargin < 2 || isempty(End) || End<=0 || End>75285
   End = 20000; % default value
end

aktif=xlsread('ActivePower');
% aktif(aktif<0)=0
reaktif=xlsread('ReactivePower');
% reaktif(reaktif<0)=0
voltage=xlsread('Voltage');
% voltage(voltage<0)=0

aktif=aktif(Start:End,1);
reaktif=reaktif(Start:End,1);
voltage=voltage(Start:End,1);

data_orj=[aktif reaktif voltage];
% remove NaN from original dataset...
data_cleaned = data_orj(~any(isnan(data_orj), 2), :);

nSample=size(data_cleaned,1);

input = data_cleaned(:,1:2);
output = data_cleaned(:,end);

pTrain=0.7;
nTrain=round(pTrain*nSample);

%% ploting dataset...

    figure(111)
    subplot(311)
    plot(input(:,1),'-.k','LineWidth',2);
    xlabel('samples')
    ylabel('Active Power')
    subplot(312)
    plot(input(:,2),'--b','LineWidth',2);
    xlabel('samples')
    ylabel('Reactive Power')
    subplot(313)
    plot(output,'-g','LineWidth',2);
    xlabel('samples')
    ylabel('Voltage')
    
% % % %% data normalization...
    Min = min(data_orj);
    Max = max(data_orj);
% % % 	
% % %     input = (input - Min(1:2))./(Max(1:2)- Min(1:2));
% % %     output = (output - Min(end))./(Max(end)- Min(end));
       
%% prepare dataset for training and test...
    N = 1;  
    % % % 	input = lagmatrix(input(:,1:2),[N:-1:0]);
    % % % 	input(isnan(input)) = 0; % NANs larin yerine 0 koy
    input3 = lagmatrix(output,N);
	input3(isnan(input3)) = 0; % NANs larin yerine 0 koy
    input = [input input3];
    TrainInputs=input(N+1:nTrain,:);
    TrainTargets=output(N+1:nTrain);
    TestInputs = input(nTrain+1:end,:);
    TestTargets = output(nTrain+1:end);

    data_output.TrainInputs=TrainInputs;
    data_output.TrainTargets=TrainTargets;
    data_output.TestInputs=TestInputs;
    data_output.TestTargets=TestTargets;
    data_output.min=Min;
    data_output.max=Max;
end