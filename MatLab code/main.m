function main
    clc; clear;

    % loads data from opinion-lexicon-English
    data = readLexicon;

    % split data into training and testing (10%)
    numWords = size(data,1);
    cvp = cvpartition(numWords,'HoldOut',0.1);
    dataTrain = data(training(cvp),:);
    dataTest = data(test(cvp),:);

    sentimentClassifier = SentimentClassifier;
    sentimentClassifier.Train(dataTrain);

    % CLASSIFIER TESTING
    % convert testing data words into word vectors using word2vec
    wordsTest = dataTest.Word;
    XTest = word2vec(sentimentClassifier.FTWEmbedding, wordsTest);
    YTest = dataTest.Label;

    sentimentClassifier.Test(XTest);

    % visualize the classification
    figure
    confusionchart(YTest,YPred);

    % TODO: Text preprocessing
end