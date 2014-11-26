classdef SMOClassifier < handle
    %SMOCLASSIFIER Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        % Primal parameters
        w;
        b;
        
        % dual parameters
        alpha;
        SVs;
        r;
        C;
        
        % kernel option
        kernel;
        
        % normalization constants
        trainMean;
        trainScale;
        
        % cost
        trainingTime;
        paramTuningTime;
    end
    
    methods
        function classifierObj = SMOClassifier(wV, bV, alphaV, CV, X, y, kernelOpt, time1, time2)
            classifierObj.w = wV;
            classifierObj.b = bV;
            
            classifierObj.trainMean = sum(X, 1)/size(X, 1);
            classifierObj.trainScale = max(sqrt( sum(X.^2, 2) ));
            X = (X - ones(size(X, 1), 1) * classifierObj.trainMean)/classifierObj.trainScale;
            
            classifierObj.alpha = sparse(alphaV);
            classifierObj.SVs = X(alphaV > 0, :);
            classifierObj.r = y(alphaV > 0);
            classifierObj.C = CV;
            classifierObj.kernel = kernelOpt;
            
            classifierObj.trainingTime = time1;
            classifierObj.paramTuningTime = time2;
        end
        function y = predict(obj, X)
            if size(X, 1) == length(obj.w)
                X = X';
            end
            if size(X, 2) ~= 2
                error('Input space dimension mismatched.');
            end
            N = size(X, 1);
            X = (X - ones(N, 1) * obj.trainMean)/obj.trainScale;
            %y = sign(X*obj.w + obj.b);
            alp = full( obj.alpha(obj.alpha > 0) );
            y = sign( model.classify.SMOClassifier.kernelEval(X, obj.SVs, obj.kernel, obj.trainScale*.3) * (obj.r.*alp) );
        end
        function [w b] = computeAffine(obj)
            % Compute an affine transformation s.t. Xw + b = predict(X)
            % Original training data have been normalized and centralized.
            % i.e. (X-mean(X))w'/k + b' = r
            %   => (X-[1; ... ; 1][xMu yMu])w'/k + [1; ... ; 1]b' = r
            %   => X(w'/k) + [1; ... ; 1](b'-[xMu yMu]w'/k) = r
            %   => X(w'/k) + (b'-[xMu yMu]w'/k) = r
            %   => w = w'/k
            %   => b = b' + [xMu yMu]w/k;
            %   => w = w' (Scaled-Invariant)
            %   => b = kb' + [xMu yMu]w;
            k = obj.trainScale;
            mean = obj.trainMean;
            w = obj.w;
            b = k*obj.b - mean*obj.w;
        end
    end
    
    methods (Static = true)
        function obj = train(inX, y, kernel, useCache, kernalCacheSize)
            % Input arguments preprocessing
            if nargin == 2
                kernel = 'gaussian';
                kernalCacheSize = 100;
                useCache = 1;
            elseif nargin == 3
                kernalCacheSize = 100;
                useCache = 1;
            elseif nargin == 4
                kernalCacheSize = 100;
            elseif nargin < 2
                error('WTF!? You should input both training set and corresponding labels.');
            end
            X = inX;
            Ntrain = length(y);
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
            
            % Kernel Cache Initialization
            cache = zeros(kernalCacheSize, 3); % [x^(t) K]
            cacheLength = 0;
            
            % Zero-mean and Scale Normalization
            gamma = max(sqrt( sum(X.^2, 2) ))*.3;
            X = X - ones(Ntrain, 1) * sum(X, 1)/Ntrain;
            X = X/max(sqrt( sum(X.^2, 2) ));
            
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
                CV = 0;
                obj = model.classify.SMOClassifier(wV, bV, alpV, CV, inX, y, 'linear', 0, 0);
                warning('There is only one group of samples.');
                return;
            end
            
            % Parameters Initialization
            start = tic;
            CV = model.classify.SMOClassifier.RANSAC(X, 100, kernel, gamma );
            time = toc(start);
            NIter = 1e5;
            A = min(0, y .* CV);
            B = max(0, y .* CV);
            hit = 0;
            tol = eps(CV * 1e2);
            idxJ = 0;
            idxI = 0;
            
            % Initial Guess
            alp = zeros(Ntrain, 1);
            grad = ones(Ntrain, 1);
            
            % Start Iteration
            for ii = 1:NIter
                if mod(ii, round(.1*NIter)) == 0
                    fprintf('%d iteration (totally %d)...\n', ii, NIter);
                end
                yg = y.*grad;
                yAlp = y.*alp;
                if idxI > 0 && idxJ > 0
                    yAlp(idxI) = 2*CV;
                    yAlp(idxJ) = -2*CV;
                end
                idx = find(yAlp < B);
                [~, idxI] = max( yg(yAlp < B) );
                idxI = idx(idxI);
                idx = find(yAlp > A);
                [~, idxJ] = min( yg(yAlp > A) );
                idxJ = idx(idxJ);
                %[idxI, idxJ];
                
                if isempty(idxJ) && isempty(idxI)
                    diff = 0;
                elseif isempty(idxJ) && ~isempty(idxI)
                    diff = -yg(idxI);
                elseif ~isempty(idxJ) && isempty(idxI)
                    diff = yg(idxJ);
                else
                    diff = yg(idxJ)-yg(idxI);
                end
                
                if (diff > -tol || ii == NIter) ... 
                        && ii > 1e1 % Stop Criterion
                    if ii == NIter
                        fprintf('Maximal Iteration Time (%d) reached.\n', NIter);
                    else
                        fprintf('Descent %d times (totally %d)\n', ii, NIter);
                    end
                    wV = X' * diag(y) * alp;
                    wV = wV/wV(2);
                    if sum(alp > 0 & alp < CV) > 0
                        freeSVsIdx = find(alp > 0 & alp < CV);
                        bV = mean( y(freeSVsIdx) - X(freeSVsIdx, :) * wV );
                    else
                        warning('There is no free support vector.');
                        bV = 0;
                    end
                    obj = model.classify.SMOClassifier(wV, bV, alp, CV, inX, y, kernel, toc(start) - time, time);
                    fprintf('Hit rate: %f\n', hit/ii/3);
                    return;
                end
                
                % Cache Implementation
                if ~isempty( find(cache(:, 1)==idxJ & cache(:, 2)==idxJ, 1) ) && useCache
                    hit = hit + 1;
                    K_jj = cache( find(cache(:, 1)==idxJ & cache(:, 2)==idxJ, 1), 3);
                else
                    K_jj = model.classify.SMOClassifier.kernelEval( ... 
                        X(idxJ, :), X(idxJ, :), kernel, gamma );
                    if cacheLength < kernalCacheSize && useCache
                        cacheLength = cacheLength + 1;
                        cache(cacheLength, :) = [idxJ idxJ K_jj];
                    elseif useCache
                        cache = [cache(2:end, :); idxJ idxJ K_jj];
                    end
                end
                
                if ~isempty( find(cache(:, 1)==idxI & cache(:, 2)==idxI, 1) ) && useCache
                    hit = hit + 1;
                    K_ii = cache( find(cache(:, 1)==idxI & cache(:, 2)==idxI, 1), 3);
                else
                    K_ii = model.classify.SMOClassifier.kernelEval( ...
                        X(idxI, :), X(idxI, :), kernel, gamma  );
                    if cacheLength < kernalCacheSize && useCache
                        cacheLength = cacheLength + 1;
                        cache(cacheLength, :) = [idxI idxI K_ii];
                    elseif useCache
                        cache = [cache(2:end, :); idxI idxI K_ii];
                    end
                end
                
                if ~isempty( find(cache(:, 1)==idxJ & cache(:, 2)==idxI, 1) ) && useCache
                    hit = hit + 1;
                    K_ij = cache( find(cache(:, 1)==idxJ & cache(:, 2)==idxI, 1), 3);
                elseif ~isempty( find(cache(:, 1)==idxI & cache(:, 2)==idxJ, 1) ) && useCache
                    hit = hit + 1;
                    K_ij = cache( find(cache(:, 1)==idxI & cache(:, 2)==idxJ, 1), 3);
                else
                    K_ij = model.classify.SMOClassifier.kernelEval( ...
                        X(idxI, :), X(idxJ, :), kernel, gamma  );
                    if cacheLength < kernalCacheSize && useCache
                        cacheLength = cacheLength + 1;
                        cache(cacheLength, :) = [idxI idxJ K_ij];
                    elseif useCache
                        cache = [cache(2:end, :); idxI idxJ K_ij;];
                    end
                end
                
                % Determin lambda
                lambda = min( [B(idxI)-y(idxI)*alp(idxI) ...
                             , y(idxJ)*alp(idxJ)-A(idxJ) ... 
                             , (yg(idxI)-yg(idxJ))/(K_ii+K_jj-2*K_ij)] );
                
                % Update gradient
                grad = grad - lambda*diag(y)* ...
                    model.classify.SMOClassifier.kernelEval(X, X(idxI, :), kernel, gamma ) ... 
                    + lambda*diag(y) * model.classify.SMOClassifier.kernelEval(X, X(idxJ, :), kernel, gamma );
                
                % Update alpha
                alp(idxI) = alp(idxI) + y(idxI)*lambda;
                alp(idxJ) = alp(idxJ) - y(idxJ)*lambda;
            end
        end
        function C = RANSAC(X, RANSACSize, k, gamma)
            % Randomly sample certain size of subset, and find an
            % appropriate C to reduce training time
            if RANSACSize > size(X, 1)
                RANSACSize = size(X, 1);
            end
            switch lower(k)
                case 'gaussian'
                    NIter = round( (1 + log(size(X, 1)/RANSACSize/2)) * 50 );
                otherwise
                    NIter = round( (1 + log(size(X, 1)/RANSACSize/2)) * 500 );
            end
            medians = zeros(NIter, 1);
            for ii = 1:NIter
                rp = randperm(size(X, 1));
                rp = rp(1:RANSACSize);
                K = model.classify.SMOClassifier.kernelEval( X(rp, :), X(rp, :), k,  gamma);
                lambda = zeros( (RANSACSize)*(RANSACSize-1)/2, 1 );
                crtIdx = 0;
                for jj = 1:RANSACSize-1
                    for kk = jj:RANSACSize
                        crtIdx = crtIdx + 1;
                        lambda(crtIdx) = .8/(K(jj, jj) + K(kk, kk) - 2*K(jj, kk))+1;
                    end
                end
                lambda = sort(lambda, 'ascend');
                lambda = lambda(1:ceil( (RANSACSize)*(RANSACSize-1)/4 ));
                medians(ii) = median(lambda);
                if RANSACSize == size(X, 1)
                    C = medians(ii);
                    return;
                end
            end
            C = round(mean(medians)*100)/100;
        end
        function v = kernelEval(x, y, opt, varargin)
            switch lower(opt)
                case 'gaussian'
                    gamma = varargin{1};
                    v = zeros(size(x, 1), size(y, 1));
                    for ii = 1:size(v, 1)
                        for jj = 1:size(v, 2)
                            v(ii, jj) = -gamma * norm( x(ii, :)-y(jj, :) );
                        end
                    end
                    v = exp(v);
                    
                case 'polynomial'
                    alp = varargin{1};
                    beta = varargin{2};
                    gamma = varargin{3};
                    v = (x*y'/alp + beta).^gamma;
                otherwise
                    v = x * y';
            end
        end
    end
    
end

