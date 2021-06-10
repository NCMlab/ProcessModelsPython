N = 100;
x = randn(N,1);
q = randn(N,1);


qES = [-2:0.05:5];
Nq = length(qES);
IE = zeros(Nq,1);

TSLS = zeros(Nq,1);
IETSLS = zeros(Nq,1);
Cerrors = zeros(Nq,1);
TrueXonM = 0.5;
TrueMonY = 0.5;
EstXonM = zeros(Nq,1);
EstMonY = zeros(Nq,1); 
for i = 1:Nq
    m = TrueXonM*x + randn(N,1) + qES(i)*q;
    y = TrueMonY*m + 1*x + randn(N,1) + qES(i)*q;
    S1 = regstats(y,x);
    S2 = regstats(m,x);
    S3 = regstats(y, [m x]);
    Cerrors(i) = corr(S2.r,S3.r);
    S4 = regstats(y, [S2.yhat]);
    IE(i) = S3.beta(2)*S2.beta(2);
    TSLS(i) = S4.beta(2);
    IETSLS(i) = (S4.beta(2) - 1/S2.beta(2))*S2.beta(2);
    EstXonM(i) = S2.beta(2);
    EstMonY(i) = S3.beta(2);
end
%
figure(1)
clf
hold on
plot(qES,IE)
% plot(qES,TSLS)
plot(qES,IETSLS)
h = line([min(qES) max(qES)],[TrueXonM*TrueMonY TrueXonM*TrueMonY]);
set(h,'Color','k')
legend('OLS', 'Two Stage','Truth')
xlabel('Size of Q effect')
title('Indirect Effects, SNR=0.5')
figure(3)
clf
hold on
plot(qES, EstXonM,'r')
line([min(qES) max(qES)],[TrueXonM TrueXonM]);
legend('Estimate','Truth')
title('Effect of X on M, SNR=0.5')
xlabel('Size of Q effect')
figure(4)
clf
hold on
plot(qES, EstMonY,'r')
line([min(qES) max(qES)],[TrueMonY TrueMonY]);
xlabel('Size of Q effect')
legend('Estimate','Truth')
title('Effect of M on Y, SNR=0.5')


%%

corr([x m y q])
S1 = regstats(y,x);
S2 = regstats(m,x);
S3 = regstats(y, [m x]);

S4 = regstats(y, [S2.yhat x]);


total = S1.beta(2)
direct = S3.beta(3)
indirect = S3.beta(2)*S2.beta(2)


MonY = S3.beta(2)

S1.beta
S2.beta

S4.beta(2)
S1.beta(2)/S2.beta(2)

%% What is the indirect effect of X on Y via Z as the joint effects of Q increase?
% Compare the indirect effect and the 2SLS



