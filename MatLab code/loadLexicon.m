% Helper function, which loads imported lexicons
function data = loadLexicon
    % Read lists of positive and negative texts from file
    file_PositiveTexts = fopen(fullfile('sentiment-lexicon','positive-texts.txt'));
    file_NegativeTexts = fopen(fullfile('sentiment-lexicon','negative-texts.txt'));
    
    textscan_PositiveTexts = textscan(file_PositiveTexts, '%s');
    textscan_NegativeTexts = textscan(file_NegativeTexts, '%s');
    
    positiveTexts = string(textscan_PositiveTexts{1});
    negativeTexts = string(textscan_NegativeTexts{1});

    fclose all; % Close all files
    
    % Create a Text:Label table
    textsAll = [positiveTexts; negativeTexts];
    labels = categorical(nan(numel(textsAll),1));
    labels(1:numel(positiveTexts)) = "Positive";
    labels(numel(positiveTexts)+1:end) = "Negative";
    
    data = table(textsAll, labels, 'VariableNames', {'Text','Label'});
end