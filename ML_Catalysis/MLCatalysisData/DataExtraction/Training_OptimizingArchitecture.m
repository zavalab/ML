clear all;
clc;

input_table = readtable('final_withoutPtAuCeO2_NoTempCorrection.txt');


%% Neural Network Training

X = table2array(input_table);
X = X' ;
numInputs = length(X);

% Neural network architectures to try:
k = 0;
for firstLayer = 5:8
    for secondLayer = 0:4
        k = k+1;
        architectures(k,1) = firstLayer; % num neurons in first layer
        architectures(k,2) = secondLayer; % num neurons in second layer
        architectures(k,3) = ((numInputs) * firstLayer) + (firstLayer * secondLayer) + secondLayer; % Total num weights
    end
end

% architectures = [[5,6,8];[1,2,0]]';

for each_arch = 1:length(architectures)
    disp("Architecture " + each_arch + " of " + length(architectures))
    % Define NN architecture
    firstLayer = architectures(each_arch,1);
    secondLayer = architectures(each_arch,2);

    if secondLayer == 0
        size = [firstLayer]; 
    else
        size = [firstLayer secondLayer];
    end
    
    for j = 1:5 % Randomly divide the data

        % Randomly assign data to training, validation and testing sets
        [dat_train,dat_test] = dividerand(X,0.8,0.2) ;

        r = length(dat_train(:,1));
            x_train = dat_train(1:r-1,:);
            y_train = dat_train(r,:);
        r = length(dat_test(:,1));
            x_test = dat_test(1:r-1,:);
            y_test = dat_test(r,:);

        % Create the Feedforward Neural Network
        net = fitnet(size, 'trainbr');
        net.trainParam.epochs = 2500;
        net.trainParam.showWindow = false; % don't display training GUI

        MSE_best = 100;
        for i = 1:5

            % Train the Neural Network
            net = init(net); % reset the initial state of the NN
            net = train(net, x_train, y_train);
            
            ypred_train = net(x_train) ;
            R_train = (corr2(y_train,ypred_train))^2;
            MSE_train = immse(y_train,ypred_train);

            % Test the Neural Network
            ypred_test = net(x_test);
            R_test = (corr2(y_test,ypred_test))^2;
            MSE_test = immse(y_test,ypred_test);
            
            if MSE_test < MSE_best
                MSE_best = MSE_test;
            end
        end
        
        MSE_bests(j) = MSE_best; % Record the best MSE of the i trainings
    end
    
    architectures(each_arch,4) = mean(MSE_bests); % Save the average "best" MSE for each architecture
    architectures(each_arch,5) = min(MSE_bests); % Save the single "best" MSE for each architecture
end


disp("DONE");
