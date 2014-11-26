clear all; close all;

load timeList;
x = (1:4) * 50;
total = zeros(3, 8);
part = timeList([1 3 5], :);

for ii = 1:3
    if ii == 1
        total(ii, :) = timeList(ii, :);
    else
        total(ii, :) = sum(timeList((ii-1)*2:ii*2-1, :));
    end
end

figure();
hold on;
errorbar(x, total(1, [1 3 5 7]), total(1, [2 4 6 8]), 'r');
errorbar(x, total(2, [1 3 5 7]), total(2, [2 4 6 8]), 'k');
errorbar(x, total(3, [1 3 5 7]), total(3, [2 4 6 8]), 'b');
hold off;
legend('CVX', 'Linear SVM', 'Gaussian SVM');
title('Total Time Consuming');
xlabel('data size');
ylabel('time (sec)');

figure();
hold on;
errorbar(x, part(1, [1 3 5 7]), part(1, [2 4 6 8]), 'r');
errorbar(x, part(2, [1 3 5 7]), part(2, [2 4 6 8]), 'k');
errorbar(x, part(3, [1 3 5 7]), part(3, [2 4 6 8]), 'b');
hold off;
legend('CVX', 'Linear SVM', 'Gaussian SVM');
title('Time Consuming without Parameters Tuning');
xlabel('data size');
ylabel('time (sec)');