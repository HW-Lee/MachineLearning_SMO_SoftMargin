import model.classify.SMOClassifier
import model.classify.SoftMarginLinearClassifier

close all; clear;

rootPath = 'ML_assignment-04-term-proj-training-data';
load(fullfile(rootPath, 'X.dat'));
load(fullfile(rootPath, 'y.dat'));

opt = 0;

fold = 20;
N = 10;

avgAccuracy = 0;
avgs = zeros(N, 1);
for jj = 1:N
    rp = randperm(length(y));
    for ii = 1:fold
        testIdx = rp( ( ii-1 )*floor( length(rp)/fold )+1:ii*floor( length(rp)/fold ) );
        X_test = X(testIdx, :);
        y_test = y(testIdx);
        trainIdx = rp;
        trainIdx( ( ii-1 )*floor( length(rp)/fold )+1:ii*floor( length(rp)/fold ) ) = [];
        X_train = X(trainIdx, :);
        y_train = y(trainIdx);
        if opt
            myClassifier = SMOClassifier.train(X_train, y_train);
        else
            myClassifier = SoftMarginLinearClassifier.train(X_train, y_train);
        end
        label = myClassifier.predict(X_test);
        fprintf('Accuracy: %f\n', sum(label==y_test)/length(y_test));
        avgAccuracy = ( avgAccuracy*(ii-1) + sum(label==y_test)/length(y_test) )/ii;
    end
    avgs(jj) = avgAccuracy;
end
fprintf('Overall Accuracy (mean) : %f\n', mean(avgs));
fprintf('Overall Accuracy (std)  : %f\n', std(avgs));