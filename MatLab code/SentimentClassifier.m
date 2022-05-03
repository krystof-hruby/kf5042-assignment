% Sentiment classifier class
% Classifies sentiment of text into Positive, Negative and Neutral
% Uses support vector machine for its model
classdef SentimentClassifier < handle
    % PRIVATE PROPERTIES
    properties (Access = private)
        FTWEmbedding
        Model
    end

    % PUBLIC PROPERTIES
    properties (Access = public)
        ClassificationAccuracy
        ClassificationPrecision
        ClassificationRecall
    end

    % PUBLIC METHODS
    methods (Access = public)
        % Constructor
        function obj = SentimentClassifier()
            % Load fastTextWordEmbedding toolbox
            % https://www.mathworks.com/help/textanalytics/ref/fasttextwordembedding.html
            obj.FTWEmbedding = fastTextWordEmbedding;
        end

        % Trains the model on parameter data
        function obj = Train(obj, data)
            % Remove words that are not included in FTWEmbedding
            removedWords = ~isVocabularyWord(obj.FTWEmbedding, data.Text);
            data(removedWords,:) = [];

            % Convert text into word-vectors using word2vec from fastTextWordEmbedding toolbox
            data_WordVectors = word2vec(obj.FTWEmbedding, data.Text);

            % Train the model for binary classification using fitcsvm
            % https://www.mathworks.com/help/stats/fitcsvm.html
            obj.Model = fitcsvm(data_WordVectors, data.Label);
        end

        % Tests the classifier on known data and shows the results as a confusion matrix and its accuracy
        function [obj, prediction] = Test(obj, data, visualize)
            % Convert words into word-vectors using word2vec from fastTextWordEmbedding toolbox
            predictionData_WordVectors = word2vec(obj.FTWEmbedding, data.Text);
            
            % Predict the sentiment
            [prediction, ~] = predict(obj.Model, predictionData_WordVectors);

            % Create confusion matrix for the classification
            confusionMatrix = confusionmat(data.Label, prediction);
            truePositives = confusionMatrix(1,1);
            trueNegatives = confusionMatrix(2,2);
            falsePositives = confusionMatrix(1,2);
            falseNegatives = confusionMatrix(2,1);
            classificationAccuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives) * 100;
            classificationPrecision = truePositives / (truePositives + falsePositives) * 100;
            classificationRecall = truePositives / (truePositives + falseNegatives) * 100;

            obj.ClassificationAccuracy = classificationAccuracy;
            obj.ClassificationPrecision = classificationPrecision;
            obj.ClassificationRecall= classificationRecall;

            % Print the confusion matrix and accuracy onto the console
            fprintf("True positives: %d\n", truePositives);
            fprintf("True negatives: %d\n", trueNegatives);
            fprintf("False positives: %d\n", falsePositives);
            fprintf("False negatives: %d\n", falseNegatives);
            fprintf("Accuracy: %d\n", round(classificationAccuracy));
            fprintf("Precision: %d\n", round(classificationPrecision));
            fprintf("Recall: %d\n", round(classificationRecall));

            % Visualize the confusion matrix
            if nargin == 3 && visualize == "visualize"
                figure
                confusionchart(data.Label, prediction);
            end
        end

        % Classifies sentiment of data
        function prediction = Classify(obj, text)
            % Convert text into word-vectors using word2vec from fastTextWordEmbedding toolbox
            data_WordVectors = word2vec(obj.FTWEmbedding, text);

            % Predict the sentiment
            [prediction,~] = predict(obj.Model, data_WordVectors);

            % Print the prediction onto the console
            fprintf("%s => %s\n", text, prediction);
        end
    end
end