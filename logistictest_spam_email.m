clear;
clc;

%% load data and labels
data   = importdata('spam_email/data.txt');
labels = importdata('spam_email/labels.txt');

%% partition data
data_train  = data(1:2000,:);
data_test   = data(2001:end,:);
labels_train = labels(1:2000);
labels_test  = labels(2001:end);

n = [200; 500; 800; 1000; 1500; 2000];
loss_gd   = zeros(length(n),1);
loss_irls = zeros(length(n),1);
for ii = 1:length(n)
    weights = logistic_train(data_train(1:n(ii),:),labels_train(1:n(ii)),"IRLS",1e-3,20);
    loss_irls(ii) = -sum(labels_test.*logsigmoid(data_test*weights)+(1-labels_test).*(-data_test*weights+logsigmoid(data_test*weights)))/length(labels_test);
    weights = logistic_train(data_train(1:n(ii),:),labels_train(1:n(ii)));
    loss_gd(ii) = -sum(labels_test.*logsigmoid(data_test*weights)+(1-labels_test).*(-data_test*weights+logsigmoid(data_test*weights)))/length(labels_test);
end

plot(n,loss_gd,'.-','LineWidth',2,'MarkerSize',20);
hold on
plot(n,loss_irls,'.-','LineWidth',2,'MarkerSize',20);
legend('gradient descent','newton-raphson','Location','NorthEast');
xlabel('size of training data n');
ylabel('logistic loss (1/0 label encoding)');
hold off

function y = logsigmoid(x)
a = max(-x,zeros(length(x),1));
y = -(log(exp(-a)+exp(-x-a))+a);
%y = -log((1+exp(-x)));
end