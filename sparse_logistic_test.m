clear;
clc;
%% load alzheimers data
load alzheimers/ad_data.mat
load alzheimers/feature_name.mat

%% perform sparse logistic regression with different regularization parameters
par = [1e-8, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
num_nonzeros = zeros(length(par),1);
test_auc = zeros(length(par),1);
legend_label = cell(length(par),1);
figure(1);
hold on
xlabel('False positive rate');
ylabel('True positive rate');
title('ROC for classification by logistic regression');
hold off
for ii = 1:length(par)
    [w , c] = logistic_l1_train(X_train, y_train, par(ii));
    scores  = X_test*w;
    [X,Y,T,AUC] = perfcurve(y_test, scores, 1);
    test_auc(ii) = AUC;
    num_nonzeros(ii) = nnz(w);
    legend_label{ii} = ['par = ' num2str(par(ii))];
    hold on
    plot(X,Y,'-','LineWidth',2);
    hold off
end
legend(legend_label,'Location','SouthEast');

figure(2);
hold on
plot(par,num_nonzeros,'r.-','LineWidth',2,'MarkerSize',20);
title('number of features selected for different regularization parameter');
xlabel('regularization parameter (par)');
ylabel('number of features selected');
hold off

figure(3);
hold on
plot(par,test_auc,'r.-','LineWidth',2,'MarkerSize',20);
title('AUC for different regularization parameter');
xlabel('regularization parameter (par)');
ylabel('AUC');
hold off