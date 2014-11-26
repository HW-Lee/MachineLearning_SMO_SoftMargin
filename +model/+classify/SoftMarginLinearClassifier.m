classdef SoftMarginLinearClassifier < handle
    %SOFTMARGINLINEARCLASSIFIER 
    %   Linear SVM Classifier with soft margin objective function.
    %   Algorithm: Projected Gradient Descent with dual problem
    
    properties
        % Primal parameters
        w;
        b;
        xi;
        
        % Dual parameters
        alpha;
        optimal_value;
        
        % Cost
        trainingTime;
    end
    
    methods
        function classifierObj = SoftMarginLinearClassifier(wV, bV, xiV, alphaV, optimalV, time1)
            classifierObj.w = wV;
            classifierObj.b = bV;
            classifierObj.xi = xiV;
            classifierObj.alpha = alphaV;
            classifierObj.optimal_value = optimalV;
            
            classifierObj.trainingTime = time1;
        end
        function y = predict(obj, X)
            if size(X, 1) == length(obj.w)
                X = X';
            end
            if size(X, 2) ~= 2
                error('Input space dimension mismatched.');
            end
            Ntrain = size(X, 1);
            X = X/max(sqrt( sum(X.^2, 2) ));
            X = X - ones(Ntrain, 1) * sum(X, 1)/Ntrain;
            y = sign(X*obj.w + obj.b);
        end
        function [w b] = computeAffine(obj, X)
            % Compute an affine transformation s.t. Xw + b = predict(X)
            % Original training data will be normalized and centralized.
            % i.e. (X-mean(X))w'/k + b' = r
            %   => (X-[1; ... ; 1][xMu yMu])w'/k + [1; ... ; 1]b' = r
            %   => X(w'/k) + [1; ... ; 1](b'-[xMu yMu]w'/k) = r
            %   => X(w'/k) + (b'-[xMu yMu]w'/k) = r
            %   => w = w'/k
            %   => b = b' + [xMu yMu]w/k;
            %   => w = w' (Scaled-Invariant)
            %   => b = kb' + [xMu yMu]w;
            if size(X, 2) ~= length(obj.w)
                if size(X, 1) == length(obj.w)
                    X = X';
                else
                    error('Input dimension mismatched.');
                end
            end
            k = max(sqrt( sum(X.^2, 2) ));
            w = obj.w;
            b = k*obj.b - sum(X, 1)/size(X, 1)*obj.w;
        end
    end
    
    methods (Static = true)
        function obj = train(X, y, C, opt, learningRate)
            start = tic;
            if nargin == 2
                C = 1e6;
                learningRate = 1e-6;
                opt = 'cvx';
            elseif nargin == 3
                learningRate = 1e-6;
                opt = 'cvx';
            elseif nargin == 4
                learningRate = 1e-6;
            elseif nargin < 2
                error('WTF!? You should input both training set and corresponding labels.');
            end
            Ntrain = length(y);
            C = C/Ntrain;
            if size(X, 1) ~= Ntrain && size(X, 2) == Ntrain
                X = X';
            end
            if size(X, 1) ~= length(y)
                error('# of training set and that of labels mismatched.');
            end
            if size(y, 1) ~= Ntrain
                y = y';
            end
            if size(y, 2) > 1
                error('Dimension of labels is unfeasible.');
            end
            
            % Zero-mean and Normalization
            X = X/max(sqrt( sum(X.^2, 2) ));
            X = X - ones(Ntrain, 1) * sum(X, 1)/Ntrain;
            
            % Check if all same labels
            if sum(y==1) == length(y) || sum(y==-1) == length(y)
                wV = [zeros(size(X, 2)-1, 1); 1];
                if sum(y==1) == length(y)
                    bV = 1-min(X(:, end));
                elseif sum(y==-1) == length(y)
                    bV = -max(X(:, end))-1;
                end
                xiV = zeros(Ntrain, 1);
                alpV = xiV;
                optimalV = 0;
                obj = model.classify.SoftMarginLinearClassifier(wV, bV, xiV, alpV, optimalV, 0);
                warning('There is only one group of samples.');
                return;
            end

            %  maximize dual function: alpha^T * c - 0.5 alpha^T * tilde(K) * alpha^T
            %  minimize: alpha^T * (0.5 tilde(K)) * alpha - c^T * alpha
            %          = alpha^T *       Q         * alpha - c^T * alpha
            %    Del   = 2 * Q * alpha^T - c
            
            % Parameters Initialization
            epsilon = 1e-2;
            NIter = max(round(.01/learningRate), 100);
            c = ones(Ntrain, 1);
            Q = .5 * (diag(y)*X) * (diag(y)*X)';
            G = [eye(Ntrain); -eye(Ntrain); ([1; -1] * y')];
            h = [( C * ones(Ntrain, 1) ); ( zeros(Ntrain+2, 1) )];
            
            if strcmpi('cvx', opt) == 1
                obj = model.classify.SoftMarginLinearClassifier.train_cvx(X, y, Q, G, h, start);
                return;
            end
            
            % Initail Guess
            %alp = ( 2*[sum(Q(y==1, :), 1); ...
            %           sum(Q(y==-1, :), 1)] ) \ ones(2, 1);
            alp = (2*Q)\c;
            h(2*Ntrain+1:end) = 1e-10;
            
            % Start Iteration
            for ii = 1:NIter
                if mod(ii, 1000) == 0
                    fprintf('%d iteration (totally %d)...\n', ii, NIter);
                end
                d = -2 * Q * alp + c;
                eta = learningRate;
                alp_next = alp + eta*d;
                isFeasible = abs(G * alp_next) <= h;
                %alp'*Q*alp - sum(alp)
                %norm(d)
                
                for jj = 1:length(isFeasible)
                    if isFeasible(jj) == 0 && jj <= 2*Ntrain
                        idx = mod(jj-1, Ntrain) + 1;
                        if alp_next(idx) < 0
                            alp_next(idx) = 0;
                        elseif alp_next(idx) > C
                            alp_next(idx) = C;
                        end
                    elseif isFeasible(jj) == 0 && jj > 2*Ntrain % Zero-sum Constrain Violated
                        sumConst = sum(y .* alp_next);
                        idx = find(y == -sign(sumConst));
                        idx = idx(alp_next(idx) < C);
                        idx(alp_next(idx) == C) = [];
                        idx1 = idx(d(idx) > 0);
                        
                        idx = find(y == sign(sumConst));
                        idx = idx(alp_next(idx) > 0);
                        idx(alp_next(idx) == 0) = [];
                        idx2 = idx(d(idx) < 0);
                        idx = [idx1; idx2];

                        maxLength = 5;
                        [~, perm] = sort(abs(d(idx)), 'descend');
                        idx = idx(perm);
                        if length(idx) > maxLength
                            idx = idx(1:maxLength);
                        end
                        alp_next(idx) = alp_next(idx) + d(idx)/sum(abs(d(idx)))*sumConst;
                    end
                end
                % Check for convergence criterion
                if norm(Q * (alp-alp_next)) < epsilon || ii == NIter % changing rate
                    if ii == NIter
                        fprintf('Maximal Iteration Time (%d) reached.\n', NIter);
                    else
                        fprintf('Descent %d times (totally %d)\n', ii, NIter);
                    end
                    alp = alp_next;
                    wV = X' * diag(y) * alp;
                    if sum(alp > 0 & alp < C) > 0
                        freeSVsIdx = find(alp > 0 & alp < C);
                        bV = mean( y(freeSVsIdx) - X(freeSVsIdx, :) * wV );
                    else
                        warning('There is no free support vector.');
                        bV = 0;
                    end
                    bV = bV/wV(2);
                    wV = wV/wV(2);
                    xiV = 1 - (y .* (X * wV + bV));
                    xiV(xiV < 0) = 0;
                    optimalV = alp' * Q * alp - sum(alp);
                    %optimalV
                    obj = model.classify.SoftMarginLinearClassifier(wV, bV, xiV, alp, optimalV, toc(start));
                    return;
                else
                    alp = alp_next;
                end
            end
        end
        function obj = train_cvx(X, y, Q, G, h, start)
            N = length(y);
            cvx_begin
                variable alp(N)
                minimize ( alp'*Q*alp - sum(alp) )
                subject to
                    G*alp <= h
            cvx_end
            wV = X' * diag(y) * alp;
            if sum(alp > 0 & alp < h(1)) > 0
                freeSVsIdx = find(alp > 0 & alp < h(1));
                bV = mean( y(freeSVsIdx) - X(freeSVsIdx, :) * wV );
            else
                warning('There is no free support vector.');
                bV = 0;
            end
            bV = bV/wV(2);
            wV = wV/wV(2);
            xiV = 1 - (y .* (X * wV + bV));
            xiV(xiV < 0) = 0;
            optimalV = alp' * Q * alp - sum(alp);
            obj = model.classify.SoftMarginLinearClassifier(wV, bV, xiV, alp, optimalV, toc(start));
        end
        function projVec = proj(theta, origVec)
            if size(theta, 1) == 1
                theta = theta';
            end
            if size(origVec, 1) == 1
                origVec = origVec';
            end
            Q = theta * pinv(theta'*theta) * theta';
            P = eye(length(Q)) - Q;
            projVec = P * origVec;
        end
    end
    
end

