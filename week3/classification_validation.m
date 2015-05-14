function [correct, incorrect] = classification_validation( x1, n1, x2, n2, c )
%CORRECT_CLASSİFİCATİON_COUNT Summary of this function goes here
%   Detailed explanation goes here
if nargin == 5
x_b=x1(1,:);
y_b=x1(2,:);
[in, on] = inpolygon(x_b, y_b, c(1,:), c(2,:));
plot(x_b(~in),y_b(~in),'bo') % blue points that are outside
incorrect_b = cumsum(in) + cumsum(on);
correct_b = n1 - incorrect_b;
    
x_r=x2(1,:);
y_r=x2(2,:);
[in, on] = inpolygon(x_r, y_r, c(1,:), c(2,:));
plot(x_r(in),y_r(in),'ro') % red points inside
correct_r = cumsum(in) + cumsum(on);
incorrect_r = n2 - correct_r

correct = correct_r(end) + correct_b(end)
incorrect = incorrect_b(end) + incorrect_r(end)
end
if nargin == 3
x1 = x1';
x = x1(1,:);
y = x1(2,:);
c = x2;
[in, on] = inpolygon(x, y, c(1,:), c(2,:));
%test labeled as 1
correct = cumsum(in) + cumsum(on);
correct = correct(end);
incorrect = n1 - correct
end
end

