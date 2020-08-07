%{
    This script performs a sensitivity analysis on the large ANN trained
    without the presence of Au/CeO2 and Pt/CeO2. The sensitivity analysis is
    done by removing each input parameter, individually, and then
    evaluating the effect this has on the error analysis or a re-trained
    network.
%}

clear all;
clc;

data_table = readtable('final_withoutPtAuCeO2_WithoutTempCorrection.txt');

AttributeRemoved = [];
SensitivityResults = [];

% Remove each column one by one

for i=1:18
    i
   
    data = data_table;
    
            if (i == 1)
                    data.BE_C=[];
                    AttributeRemoved = [AttributeRemoved "Binding Energy Carbon"];
            elseif (i == 2) 
                    data.loading_base=[];
                    AttributeRemoved = [AttributeRemoved "Primary Metal Loading, wt%"];
            elseif (i == 3) 
                    data.Z_IonicRad=[];
                    AttributeRemoved = [AttributeRemoved "Promoter Charge/Ionic Radius"];
            elseif (i == 4)  
                    data.Electronegativity=[];
                    AttributeRemoved = [AttributeRemoved "Promoter Electronegativity"];
            elseif (i == 5) 
                    data.loading_promoter=[];
                    AttributeRemoved = [AttributeRemoved "Promoter Loading, wt%"];
            elseif (i == 6) 
                    data.FirstIE_supp=[];
                    AttributeRemoved= [AttributeRemoved "Support, First Ionization Energy"];
            elseif (i == 7) 
                    data.Electroneg_supp=[];
                    AttributeRemoved = [AttributeRemoved "Support, Electronegativity"];
            elseif (i == 8)  
                    data.IWI=[];
                    data.WI=[];
                    data.CI=[];
                    data.SI=[];
                    data.HDP=[];
                    data.FSP=[];
                    data.DP=[];
                    AttributeRemoved = [AttributeRemoved "Synthesis"];
            elseif (i == 9) 
                    data.CalcT_C =[];
                    AttributeRemoved = [AttributeRemoved "Calcination Temperature, C"];
            elseif (i == 10)  
                    data.CalcT_time =[];
                    AttributeRemoved = [AttributeRemoved "Calcination Time"];
            elseif (i == 11) 
                    data.T_K =[];
                    AttributeRemoved = [AttributeRemoved "Reaction Temperature"];
            elseif (i == 12) 
                    data.H2Vol_=[];
                    AttributeRemoved = [AttributeRemoved "Reactant Feed, H2 vol%"];
            elseif (i == 13)  
                    data.COVol_ =[];
                    AttributeRemoved = [AttributeRemoved "Reactant Feed, CO vol%"];
            elseif (i == 14)  
                    data.H2OVol_=[];
                    AttributeRemoved = [AttributeRemoved "Reactant Feed, H2O vol%"];
            elseif (i == 15) 
                    data.CO2Vol_=[];
                    AttributeRemoved = [AttributeRemoved "Reactant Feed, CO2 vol%"];
            elseif (i == 16) 
                    data.TOS_min_=[];
                    AttributeRemoved = [AttributeRemoved "Time on Stream (min)"];
            elseif (i == 17)
                    data.F_W =[];
                    AttributeRemoved = [AttributeRemoved "Total Inlet Flowrate/Catalyst Mass"];
            elseif (i == 18)
                    % Base Case
                    % Don't Remove any attributes
                    AttributeRemoved = [AttributeRemoved "No Attributes Removed, Base Case"];
            end
                    
    data = table2array(data);

    % Randomly assign data to training, validation and testing sets
    data = data';

    for j=1:5 % Number of iterations

        % Randomly assign data to training, validation and testing sets
        [dat_train,dat_test] = dividerand(data,0.8,0.2) ;

        r = length(dat_train(:,1));
            x_train = dat_train(1:r-1,:);
            y_train = dat_train(r,:);
        r = length(dat_test(:,1));
            x_test = dat_test(1:r-1,:);
            y_test = dat_test(r,:);


        % Create the Feedforward Neural Network
        size = [6 2];  
        net = fitnet(size, 'trainbr');
        net.trainParam.epochs = 2500;
        net.trainParam.showWindow = false; % don't display training GUI

        MSE_best = 100;
        R_best = 0.5;
        for k = 1:5
            
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
            
            if R_test > R_best
                R_best = R_test;
            end
        end
        
        MSE_bests(j) = MSE_best; % Record the best MSE, R^2 of the k trainings
        R_bests(j) = R_best;

    end 

    SensitivityResults(i,1) = mean(MSE_bests); % Record Best MSE
    SensitivityResults(i,2) = mean(R_bests); % Record Best R^2
end

Results = table(AttributeRemoved', SensitivityResults);