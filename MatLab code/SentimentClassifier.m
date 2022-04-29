classdef SentimentClassifier < handle
    properties
        FTWEmbedding
        Model
    end

    methods
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

        function prediction = Test(data)
            % Predict sentiment 
            [prediction,~] = predict(sentimentClassifier.Model, data);
        end
    end
end