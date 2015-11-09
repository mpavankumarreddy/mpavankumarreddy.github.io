rng('default');
rng('shuffle');
%rng(2);

n = 2000;
p_t = 0.75;

wrapS = @(x, s) (1 + mod(x-1, s));

alpha = 6;
beta = 2;

unif = rand(n,1);
% converting uniform distribution to 1 and -1 with probability p_t
tasks = (unif <= p_t)*2 - 1;


graph = zeros(n, 1);
A = zeros(n, 1);

graph = [];
A = [];
reliability = [];



nSub = 400;
for taskSet = 1:n/nSub

%nStart = 1;

nStart = (taskSet-1)*nSub + 1;

% Set of 5 workers to be tested out of which 2 could be chosen.
for workerSet = 1:3
m = 5;

l = 5;

% reliablity of workers sampled from beta distribution
p = 0.1 + 0.9*betarnd(alpha,beta,m,1);%0.9*ones(m,1);

tasksSub = tasks(nStart:nStart + nSub - 1);

graphSub = zeros(nSub, m);
ASub = zeros(nSub, m);

% A complete graph generation
for i = 1:nSub
  %indicesN = wrapS(i:i+l-1, n);
  indicesM = wrapS(i:i+l-1, m);
  graphSub(i, indicesM) = ones(l, 1);
  ASub(i, indicesM) = ((rand(l,1) <=  p(indicesM))*2 - 1)*tasks(i);
end

% q should mu_p(t|A). The columns correspond to {-1, 1}
q = zeros(nSub, 2);
% initialisation
for i = 1:nSub
  q(i, 1) = sum(ASub(i, :) == -1)/sum(graphSub(i, :) == 1);
  q(i, 2) = sum(ASub(i, :) == 1)/sum(graphSub(i, :) == 1);
end

pj = zeros(1, m);
for j = 1:m
  dj = find(graphSub(:, j) == 1);
  pj(j) = sum(q(dj + ((ASub(dj,j) + 3)/2-1)*nSub))/numel(dj);
end


for k = 1:50
  % E-step
  p_vals = [1-pj;pj]';
  for i = 1:nSub
    di = find(graphSub(i, :) == 1);
    ti = 1;
    %prod_plus = prod(p_vals((A(i, di) == ti)*m + di));
    prod_plus = p_t*prod(p_vals((ASub(i, di) == ti)*m + di));
    ti = -1;
    %prod_minus = prod(p_vals((A(i, di) == ti)*m + di));
    prod_minus = (1-p_t)*prod(p_vals((ASub(i, di) == ti)*m + di));
    q(i, 1) = prod_minus / (prod_minus + prod_plus);
    q(i, 2) = prod_plus / (prod_minus + prod_plus);
  end

  % M-step
  for j = 1:m
    dj = find(graphSub(:, j) == 1);
    pj(j) = sum(q(dj + ((ASub(dj,j) + 3)/2-1)*nSub))/numel(dj);
  end
end

% change criterion and test later
best = find(pj>0.88);
remaining = pj(find(pj <= 0.9));
[top topIdx] = sort(remaining);
if numel(best) < 2
  if top(end) > 0.775
    best = [best find(pj == top(end))];
  end
end

[p pj'];

pj = pj(best);
p = p(best);
graphSub = graphSub(:, best);
ASub = ASub(:, best);
m = numel(best);
[p pj'];

reliability = [reliability; [p pj']];

graph(nStart:nStart+nSub-1, end+1: end+m) = graphSub;
A(nStart:nStart+nSub-1, end+1: end+m) = ASub;
end
end



pj = reliability(:, 2)';
m = numel(pj);

q = zeros(n, 2);
% initialisation
for i = 1:n
  q(i, 1) = sum(A(i, :) == -1)/sum(graph(i, :) == 1);
  q(i, 2) = sum(A(i, :) == 1)/sum(graph(i, :) == 1);
end


display('just majority voting')
[row_max row_argmax] = max( q, [], 2 );
current_predictions = (row_argmax*2 - 3);
successful = sum(tasks == current_predictions)
total = n


%{
pj = zeros(1, m);
for j = 1:m
  dj = find(graph(:, j) == 1);
  pj(j) = sum(q(dj + ((A(dj,j) + 3)/2-1)*n))/numel(dj);
end
%}

% Log-likelihood function
% L(pj)


for k = 1:50
  % E-step
  p_vals = [1-pj;pj]';
  for i = 1:n
    di = find(graph(i, :) == 1);
    ti = 1;
    %prod_plus = prod(p_vals((A(i, di) == ti)*m + di));
    prod_plus = p_t*prod(p_vals((A(i, di) == ti)*m + di));
    ti = -1;
    %prod_minus = prod(p_vals((A(i, di) == ti)*m + di));
    prod_minus = (1-p_t)*prod(p_vals((A(i, di) == ti)*m + di));
    q(i, 1) = prod_minus / (prod_minus + prod_plus);
    q(i, 2) = prod_plus / (prod_minus + prod_plus);
  end

  % M-step
  for j = 1:m
    dj = find(graph(:, j) == 1);
    pj(j) = sum(q(dj + ((A(dj,j) + 3)/2-1)*n))/numel(dj);
  end
end

display('running done')
[row_max row_argmax] = max( q, [], 2 );
c = (row_argmax*2 - 3);
successful = sum(tasks == c)
total = n