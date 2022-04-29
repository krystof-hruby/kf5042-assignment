function main
    % Clear workspace and console
    clc; clear;

    % TODO:
    % loads data from opinion-lexicon-English
    data = readLexicon;

    % split data into training and testing (10%)
    numWords = size(data,1);
    cvp = cvpartition(numWords,'HoldOut',0.1);
    dataTrain = data(training(cvp),:);
    dataTest = data(test(cvp),:);
    % TODO END

    % Create a sentiment classifier (class)
    sentimentClassifier = SentimentClassifier;
    
    % Train the sentiment classifier
    sentimentClassifier.Train(dataTrain);

    % Test the sentiment classifier
    % Calculates the confusion matrix and accuracy of the trained model
    sentimentClassifier.Test(dataTest.Word, dataTest.Label, "visualize");

    % Example use of the classifier (can be used once trained):
    textData = "Trash, shit, bad, horrible";
    sentimentClassifier.Classify(textData);
end
























