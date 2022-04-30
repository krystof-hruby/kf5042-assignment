% Splits data into training and testing sets
% testingAmount decides how much data should be left out for testing (in %)
function [data_Training, data_Testing] = splitData(data, testingAmount)
    dataPartition = cvpartition(size(data, 1), 'HoldOut', testingAmount);
    
    data_Training = data(training(dataPartition),:);
    data_Testing = data(test(dataPartition),:);
end