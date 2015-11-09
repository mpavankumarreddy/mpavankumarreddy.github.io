n_part = [1 2 5 10 100 200 1000 2000];

error = 1 - [1980 1985 1992 1994 1991 1992 1983 1986]./2000;

q_correct = [0.99034 0.99021 0.99574 0.99789 0.997447 0.99999 1.0000 1.0000];

q_wrong = [0.4573328 5.000469e-01 0.6273 0.76723 0.95974 0.997645 1.000 1.000];

figure(1)
semilogx(n_part*15, smooth(error), '-+')
xlabel('No. of Workers (m)')
title('Non Adaptive EM with l=15')
ylabel('Avg. Probability of Error')

figure(2)
semilogx(n_part*15, q_correct, 'o-b')
hold on
semilogx(n_part*15, q_wrong, '--*r')
xlabel('No. of Workers (m)')
title('Non Adaptive EM with l=15')
ylabel('Avg. Quality q = |2p(+1) -1|')
ylim([0 1.5])
legend('correct tasks','wrong tasks')