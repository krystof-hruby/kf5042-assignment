accuracy = 0;
precision = 0;
recall = 0;


for index = 1:10
    % Clear workspace and console
    %clc; clear;
    
    % Load training data from lexicons into the workspace
    data = loadLexicon;
    
    % Split data into training and testing sets
    testingAmount = 0.05; % 5% reserved for testing
    [data_Training, data_Testing] = splitData(data, testingAmount);

    % Create a sentiment classifier (class)
    sentimentClassifier = SentimentClassifier;
    
    % Train the sentiment classifier
    sentimentClassifier.Train(data_Training);
    
    % Test the sentiment classifier
    % Calculates the confusion matrix and accuracy of the trained model
    sentimentClassifier.Test(data_Testing, "novisualize");
    
    accuracy = accuracy + round(sentimentClassifier.ClassificationAccuracy);
    precision = precision + round(sentimentClassifier.ClassificationPrecision);
    recall = recall + round(sentimentClassifier.ClassificationRecall);

    % Example use of the classifier (can be used once trained):
    textData = "The product was not very good.";
    sentimentClassifier.Classify(textData);
end

fprintf("%d \n", round(accuracy / 10));
fprintf("%d \n", round(precision / 10));
fprintf("%d \n", round(recall / 10));