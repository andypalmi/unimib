
clear; clc;
addpath 'functions'

GROUP_0 = 'negativi';
GROUP_1 = 'positivi';

% Loading data
[data, labels] = loading__fromexcel();
data0 = data(labels == 0, :);
data1 = data(labels == 1, :);

N0 = size(data0, 1);
N1 = size(data1, 1);

% Data matrix + labels'
data0 = double(data0);
data1 = double(data1);
data = double([data0; data1]);
labels = [zeros(1, N0) ones(1, N1)];
clear data0; clear data1;

% Sample information
original_indices = 1:(N0+N1);
NNeg = N0;
NPos = N1;

% Training and parameter tuning
kernel = 'linear';
K = 5;
numvar = size(data, 2);

% Cross-validation index
indices_1 = crossvalind('Kfold', N1, K); %GROUP_1
indices_0 = crossvalind('Kfold', N0, K); %GROUP_0
index = [indices_0; indices_1];

% Metrics
accuracy = 0;
sensitivity = 0;
specificity = 0;

for ind = 1:K
    disp(['K = ' num2str(ind)]);
    
    temp_train_set = true(1,size(data,1));
    temp_train_set(index == ind) = false;
    temp_test_set =~temp_train_set;

    temp_train_indices = original_indices(temp_train_set);
    temp_test_indices = original_indices(temp_test_set);

    temp_train_labels = labels(temp_train_indices);
    temp_test_labels = labels(temp_test_indices);
    
    temp_train_data = data(temp_train_indices,:);
    temp_test_data = data(temp_test_indices,:); 

    try

        pc_tr_data = squeeze(temp_train_data(:,1:numvar));
        svmStruct = fitcsvm(pc_tr_data, temp_train_labels,...
            'KernelFunction', kernel, 'Standardize', true, ...
            'BoxConstraint', 1);
        svmStruct = compact(svmStruct);

    catch exception

        % disp('Error optimization preprocessing');
        msgString = getReport(exception)

    end

    % TESTING phase
    for subject = 1:size(temp_test_data, 1)

        % Testing data
        temp_tdata = temp_test_data(subject,:);
        temp_tlabel = temp_test_labels(subject);

        te_data = temp_tdata;

        if size(te_data,2) == 0
            
            disp('No data!');
        
        else

            pc_te_data = squeeze(te_data(:,1:numvar));

            class = predict(svmStruct, pc_te_data)';
            class = cast(class,'double');
            clear pc_te_data;

            % Caso 0 -> Negative
            if temp_tlabel == 0 && class == 0
                accuracy = accuracy + (1/size(data, 1));
                specificity = specificity + (1/NNeg);
            end

            % Caso 1 -> Positive
            if temp_tlabel == 1 && class == 1
                accuracy = accuracy + (1/size(data, 1));
                sensitivity = sensitivity + (1/NPos);
            end
           
        end

        clear te_data

    end
        
end % cv

disp('.');
disp('.');
disp('.');
disp('Results');
disp('-----');
disp(['Accuracy: ' num2str(accuracy)]);
disp(['Sensitivity: ' num2str(sensitivity)]);
disp(['Specificity: ' num2str(specificity)]);
