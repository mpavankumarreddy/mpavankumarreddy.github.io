a = [
    0.2278
    0.2152
    0.2297
    0.2368
    0.1911
    0.2282
    0.2173
    0.0090
    0.0052
    0.0036]

b =[0.0015
    0.0013
    0.0007
    0.0003]

c =[0.0007
    0.0009
    0.0014
    0.0006]

d =[0.0021
    0.0011
    0.0007
    0.0006]

% l=15
% ((1.875*10^-4+0.0045/40)*40 + 0.45*10^-3*20 + 0.0003*20 + 2.2500e-04*20)/100 = 3.1500e-04

x = [a; (b*0.25+c*0.25+d*0.5); 3.1500e-04]

semilogy(smooth(x))
semilogy(x)
xlabel('l')
title('Error VS budget(l)')
ylabel('Avg. Probability of Error')
hold on
semilogy(15, x(end), 'blacko')
legend('avg. error','l=15 error=0.000333')