import model.classify.SoftMarginLinearClassifier
import model.classify.SMOClassifier

close all; clear;

rootPath = 'ML_assignment-04-term-proj-training-data';
load(fullfile(rootPath, 'X.dat'));
load(fullfile(rootPath, 'y.dat'));

k = 200;
N = 100;

avgs = zeros(N, 5);
for jj = 1:N
    if mod(jj, 10) == 0
        fprintf('N = %d\n', jj);
    end
    rp = randperm(length(y));
    X_train = X(rp(1:k), :);
    y_train = y(rp(1:k));
    myClassifier = SoftMarginLinearClassifier.train(X_train, y_train);
    avgs(jj, 1) = myClassifier.trainingTime;
    myClassifier = SMOClassifier.train(X_train, y_train, 'linear');
    avgs(jj, 2:3) = [myClassifier.paramTuningTime myClassifier.trainingTime];
    myClassifier = SMOClassifier.train(X_train, y_train, 'gaussion');
    avgs(jj, 4:5) = [myClassifier.paramTuningTime myClassifier.trainingTime];
end
meanTime = mean(avgs, 1);
stdTime = std(avgs, 1);
fprintf('Training Time (SoftMargin)  :  %.3f +- %.3f\n', meanTime(1), stdTime(1));
fprintf('Training Time (Linear SVM)  : (%.3f +- %.3f) + (%.3f +- %.3f)\n' ... 
    , meanTime(2), stdTime(2), meanTime(3), stdTime(3));
fprintf('Training Time (Gaussian SVM): (%.3f +- %.3f) + (%.3f +- %.3f)\n' ... 
    , meanTime(4), stdTime(4), meanTime(5), stdTime(5));
