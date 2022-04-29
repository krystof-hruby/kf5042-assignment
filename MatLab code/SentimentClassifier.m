classdef SentimentClassifier < handle
    % PRIVATE PROPERTIES
    properties (Access = private)
        FTWEmbedding
        Model
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
            removedWords = ~isVocabularyWord(obj.FTWEmbedding, data.Word);
            data(removedWords,:) = [];

            % Convert words into word-vectors using word2vec from fastTextWordEmbedding toolbox
            data_WordVectors = word2vec(obj.FTWEmbedding, data.Word);
            data_Labels = data.Label;

            % Train the SVM for binary classification using fitcsvm
            % https://www.mathworks.com/help/stats/fitcsvm.html
            obj.Model = fitcsvm(data_WordVectors, data_Labels);
        end

        % Tests the classifier on known data and shows the results as a confusion matrix
        function prediction = Test(obj, predictionData, trueData, visualize)
            % Convert words into word-vectors using word2vec from fastTextWordEmbedding toolbox
            predictionData_WordVectors = word2vec(obj.FTWEmbedding, predictionData);
            
            % Predict the sentiment
            [prediction, ~] = predict(obj.Model, predictionData_WordVectors);
            
            % Create confusion matrix for the classification
            confusionMatrix = confusionmat(trueData, prediction);
            truePositives = confusionMatrix(1,1);
            trueNegatives = confusionMatrix(2,2);
            falsePositives = confusionMatrix(1,2);
            falseNegatives = confusionMatrix(2,1);
            classificationAccuracy = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives) * 100;
            
            % Print the confusion matrix onto the console
            fprintf("True positives: %d\n", truePositives);
            fprintf("True negatives: %d\n", trueNegatives);
            fprintf("False positives: %d\n", falsePositives);
            fprintf("False negatives: %d\n", falseNegatives);
            fprintf("Accuracy: %d%%\n",round(classificationAccuracy));

            % Visualize the confusion matrix
            if nargin == 4 && visualize == "visualize"
                figure
                confusionchart(trueData, prediction);
            end
        end

        % 
        function prediction = Classify(obj, data)
            % Preprocess the text
            processedData = obj.PreprocessText(data);

            % Remove words that are not included in FTWEmbedding
            removedWords = ~isVocabularyWord(obj.FTWEmbedding, processedData.Vocabulary);
            processedData = removeWords(processedData, removedWords);

            % Transform tokens into strings
            processedData_Strings = string(processedData);

            % Convert words into word-vectors using word2vec from fastTextWordEmbedding toolbox
            processedData_WordVectors = word2vec(obj.FTWEmbedding, processedData_Strings);

            % Predict sentiments for each word
            [~,scores] = predict(obj.Model, processedData_WordVectors);
            
            % Calculate sentiment for the whole text (using mean)
            sentimentScore = mean(scores(:,1));

            % Decide the prediction based on calculated score and print it onto the console
            if sentimentScore == 0
                prediction = "Neutral";
                fprintf("%s => %s (0)\n", data, prediction);
            elseif sentimentScore > 0
                prediction = "Positive";
                fprintf("%s => %s (+%.2f)\n", data, prediction, sentimentScore);
            else
                prediction = "Negative";
                fprintf("%s => %s (%.2f)\n", data, prediction, sentimentScore);
            end
        end
     end

     % PRIVATE METHODS
     methods(Access = private)   
        % Preprocess text for classification
        function processedText = PreprocessText(obj, text)
            % Tokenize the text (split text into word tokens)
            processedText = tokenizedDocument(text);
            
            % Remove punctuation
            processedText = erasePunctuation(processedText);
            
            % Remove stop words
            processedText = removeStopWords(processedText);
            
            % Convert all letters to lowercase
            processedText = lower(processedText);
        end
    end
end