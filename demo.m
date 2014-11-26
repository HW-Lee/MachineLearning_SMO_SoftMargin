import model.classify.SoftMarginLinearClassifier
import model.classify.SMOClassifier

close all; clear;

rootPath = 'ML_assignment-04-term-proj-training-data';
load(fullfile(rootPath, 'X.dat'));
load(fullfile(rootPath, 'y.dat'));

SMLC = 0;
SMOC = 1;
opt = 1;

%%% TODO: please import the dataset here %%%

%%% TODO: modify the DummyClassifier to your SoftMarginLinearClassifier %%%
%%%       please follow the specs strickly                              %%%

C = 1e6;
if opt == SMLC
    myClassifier = SoftMarginLinearClassifier.train(X, y, C, 'custom');
elseif opt == SMOC
    tic;
    myClassifier = SMOClassifier.train(X, y, 'gaussian');
    toc;
    C = myClassifier.C;
end
label1 = myClassifier.predict(X);
model{1}= myClassifier;
if opt == SMLC
    myClassifier = SoftMarginLinearClassifier.train(X, y);
    label2 = myClassifier.predict(X);
    model{2} = myClassifier;
end

%%% Check constraints
for xx = 1:length(model)
    %if abs(sum(model{xx}.alpha .* y)) > 1e-10 || ... 
    %sum(model{xx}.alpha >= 0 & model{xx}.alpha <= C) < length(y)
    if 1
        fprintf('Idx: %d\n', xx);
        fprintf('Zero sum check: %e\n', ... 
            sum( full( model{xx}.alpha(model{xx}.alpha~=0) ) .* y(model{xx}.alpha~=0) ));
        fprintf('Box constrain check: %d/%d\n', ... 
            sum(full(model{xx}.alpha >= 0 & model{xx}.alpha <= C)), length(y));
    end
end


figure();
hold on;

plot(X(label1==1,1),X(label1==1,2),'rx');
plot(X(label1==-1,1),X(label1==-1,2),'bx');
scatter (X(y==1,1),X(y==1,2),'r');
scatter(X(y==-1,1),X(y==-1,2),'b');

hold off;


fprintf('Accuracy: %f\n', sum(label1==y)/length(y));
return;
%%% plot data %%%
figure();
hold on;
xx = linspace(min(X(:, 1)), max(X(:, 1)));

if opt == SMLC
    [w b] = model{1}.computeAffine(X);
    plot(xx, -(w(1)*xx+b)/w(2), 'r');
    [w b] = model{2}.computeAffine(X);
    plot(xx, -(w(1)*xx+b)/w(2), 'k');
else
    [w b] = myClassifier.computeAffine(X);
    plot(xx, -(w(1)*xx+b)/w(2), 'r'); % --> (1)
    idx1 = find(myClassifier.alpha < myClassifier.C & myClassifier.alpha > 0);
    idx2 = find(myClassifier.alpha == myClassifier.C);
    idx = idx1;
    freeSV_p = [X(idx(y(idx)==1), 1) X(idx(y(idx)==1), 2)];
    freeSV_n = [X(idx(y(idx)==-1), 1) X(idx(y(idx)==-1), 2)];
    if ~isempty(freeSV_p)
        plot(xx, -( w(1)*(xx-freeSV_p(1, 1)) )/w(2) + freeSV_p(1, 2), 'g'); % --> (2)
    end
    if ~isempty(freeSV_n)
        plot(xx, -( w(1)*(xx-freeSV_n(1, 1)) )/w(2) + freeSV_n(1, 2), 'b'); % --> (3)
    end
    idx = idx2;
    boundedSV_p = [X(idx(y(idx)==1), 1) X(idx(y(idx)==1), 2)];
    boundedSV_n = [X(idx(y(idx)==-1), 1) X(idx(y(idx)==-1), 2)];
    plot([freeSV_p(:, 1); boundedSV_p(:, 1)], [freeSV_p(:, 2); boundedSV_p(:, 2)], 'g*'); % --> (4)
    plot([freeSV_n(:, 1); boundedSV_n(:, 1)], [freeSV_n(:, 2); boundedSV_n(:, 2)], 'b*'); % --> (5)
end
if opt == SMLC
    legend('My decision boundary', 'CVX''s decision boundary');
else
    legend('SMO decision boundary', 'SVs bound+', 'SVs bound-', 'SVs+', 'SVs-');
end
scatter (X(y==1,1),X(y==1,2),'g');
scatter(X(y==-1,1),X(y==-1,2),'b');
hold off;

if opt == SMLC
    fprintf('Accuracy1: %f\n', sum(label1==y)/length(y));
    fprintf('Accuracy2: %f\n', sum(label2==y)/length(y));
else
    fprintf('Accuracy: %f\n', sum(label1==y)/length(y));
end