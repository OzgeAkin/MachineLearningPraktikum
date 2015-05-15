clear all, close all
write_fig = 0;

n1 = 80; n2 = 40;
S1 = eye(2); S2 = [1 0.95; 0.95 1];
m1 = [0.75; 0]; m2 = [-0.75; 0];
x1 = bsxfun(@plus, chol(S1)'*gpml_randn(0.2, 2, n1), m1);        
x2 = bsxfun(@plus, chol(S2)'*gpml_randn(0.3, 2, n2), m2); 

x = [x1 x2]'; y = [-ones(1,n1) ones(1,n2)]';
figure(6)
plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on
plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12);

n_1 = 20; n_2 = 20;
S_1 = eye(2); S_2 = [1 0.85; 0.85 1];
m_1 = [0.85; 0]; m_2 = [-0.55; 0];
x_test1 = bsxfun(@plus, chol(S_1)'*gpml_randn(0.2, 2, n_1), m_1);        
x_test2 = bsxfun(@plus, chol(S_2)'*gpml_randn(0.3, 2, n_2), m_2);

%x_test = [x_test1 x_test2]';
x_test = [x_test1(:) x_test2(:)];
y_test = [-ones(1,n_1) ones(1,n_2)]';
plot(x_test1(1,:), x_test1(2,:), 'b*', 'MarkerSize', 12); hold on
plot(x_test2(1,:), x_test2(2,:), 'r*', 'MarkerSize', 12);

p_test1 = n1*exp(-sum(x_test*inv(S1).*x_test/2,2))/sqrt(det(S1));
p_test2 = n2*exp(-sum(x_test*inv(S2).*x_test/2,2))/sqrt(det(S2));

[t1 t2] = meshgrid(-4:0.1:4,-4:0.1:4);
t = [t1(:) t2(:)]; n = length(t);               % these are the test inputs
tmm = bsxfun(@minus, t, m1');
p1 = n1*exp(-sum(tmm*inv(S1).*tmm/2,2))/sqrt(det(S1));

tmm = bsxfun(@minus, t, m2');
p2 = n2*exp(-sum(tmm*inv(S2).*tmm/2,2))/sqrt(det(S2));

set(gca, 'FontSize', 24)
reshaped = reshape(p2./(p1+p2), size(t1));
%contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.1:0.1:0.9])

[c h] = contour(x_test1, x_test2, reshape(p_test2./(p_test1+p_test2), size(x_test1)), [0.5 0.5]);
%[c h] = contour(t1, t2, reshape(p2./(p1+p2), size(t1)), [0.5 0.5]);
%x_train 
%[correct, incorrect] = classification_validation(x1,n1,x2,n2,c);
%test
%[correct_1, incorrect_1] = classification_validation(t,n,c);
%x_test
[correct_1, incorrect_1] = classification_validation(x_test1,n_1,x_test2,n_2,c);
accuracy_1 = correct_1 / (correct_1 + incorrect_1)

clabel(c)
set(h, 'LineWidth', 2)
colorbar
grid
axis([-4 4 -4 4])
if write_fig, print -depsc f6.eps; end

meanfunc = @meanConst; hyp.mean = 0;
covfunc = @covSEard;   hyp.cov = log([1 1 1]);
likfunc = @likGauss;  hyp.lik = log(0.1);

hyp = minimize(hyp, @gp, -40, @infExact, meanfunc, covfunc, likfunc, x, y);
%[a b c d lp] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, t, ones(n,1));
[a b c d lp] = gp(hyp, @infExact, meanfunc, covfunc, likfunc, x, y, x_test, y_test);
figure(7)
set(gca, 'FontSize', 24)
plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on
plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12)
%contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
%[c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
[c h] = contour(x_test1, x_test2, reshape(exp(lp), size(x_test1)), [0.5 0.5]);
[correct_2, incorrect_2] = classification_validation(x_test1,n_1,x_test2,n_2,c);
%[correct_2, incorrect_2] = classification_validation(t,n,c);
accuracy_2 = correct_2 / (correct_2 + incorrect_2)
set(h, 'LineWidth', 2)
colorbar
grid
axis([-4 4 -4 4])
if write_fig, print -depsc f7.eps; end

[u1,u2] = meshgrid(linspace(-2,2,5)); u = [u1(:),u2(:)]; clear u1; clear u2
nu = size(u,1);
covfuncF = {@covFITC, {covfunc}, u};
inffunc = @infFITC_EP;                     % one could also use @infFITC_Laplace
hyp = minimize(hyp, @gp, -40, inffunc, meanfunc, covfuncF, likfunc, x, y);
%[a b c d lp] = gp(hyp, inffunc, meanfunc, covfuncF, likfunc, x, y, t, ones(n,1));
[a b c d lp] = gp(hyp, inffunc, meanfunc, covfuncF, likfunc, x, y, x_test, y_test);
figure(8)
set(gca, 'FontSize', 24)
plot(x1(1,:), x1(2,:), 'b+', 'MarkerSize', 12); hold on
plot(x2(1,:), x2(2,:), 'r+', 'MarkerSize', 12)
%contour(t1, t2, reshape(exp(lp), size(t1)), 0.1:0.1:0.9)
%[c h] = contour(t1, t2, reshape(exp(lp), size(t1)), [0.5 0.5]);
[c h] = contour(x_test1, x_test2, reshape(exp(lp), size(x_test1)), [0.5 0.5]);
[correct_3, incorrect_3] = classification_validation(x_test1,n_1,x_test2,n_2,c);
%[correct_3, incorrect_3] = classification_validation(t,n,c);
accuracy_3 = correct_3 / (correct_3 + incorrect_3)
set(h, 'LineWidth', 2)
plot(u(:,1),u(:,2),'ko', 'MarkerSize', 12)
colorbar
grid
axis([-4 4 -4 4])
if write_fig, print -depsc f8.eps; end