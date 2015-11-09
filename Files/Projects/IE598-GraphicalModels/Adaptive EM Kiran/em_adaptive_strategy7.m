%rng('default');
rng('shuffle');
%rng(2);

n = 2000;
p_t = 0.75;

wrapS = @(x, s) (1 + mod(x-1, s));

alpha = 6;
beta = 2;

l = 8;
m = 200;
%r = 2000;

%m = n*l/r;
r = n*l/m;

unif = rand(n,1);
% converting uniform distribution to 1 and -1 with probability p_t
tasks = (unif <= p_t)*2 - 1;

% reliablity of workers sampled from beta distribution
p = 0.1 + 0.9*betarnd(alpha,beta,m,1);%0.9*ones(m,1);


graph = zeros(n, m);
A = zeros(n, m);
%{
% generating a graph - adjacency matrix using configuration model
left_half = reshape(repmat(1:n,l,1),n*l,[]);
right_half = reshape(repmat(1:m,r,1),m*r,[]);

index = randperm(n*l);
right_half = right_half(index);

for i = 1:(n*l)
  graph(left_half(i), right_half(i)) = 1;
  A(left_half(i), right_half(i)) = -tasks(left_half(i));
  if (rand() <= p(right_half(i)))
    A(left_half(i), right_half(i)) = tasks(left_half(i));
  end
end
%}

%different graph generation
graph = ceil( rand(n,m)-1+(l/m) );
for i = 1:n
  di = find(graph(i, :) == 1);
  A(i, di) = ((rand(numel(di), 1) <= p(di))*2 - 1) * tasks(i);
end
%     for i = 1:n
%       %indicesN = wrapS(i:i+l-1, n);
%       indicesM = wrapS(i:i+l-1, m);
%       graph(i, indicesM) = ones(l, 1);
%       A(i, indicesM) = ((rand(l,1) <=  p(indicesM))*2 - 1)*tasks(i);
%     end


% q should mu_p(t|A). The columns correspond to {-1, 1}
q = zeros(n, 2);
% initialisation
for i = 1:n
  q(i, 1) = sum(A(i, :) == -1)/sum(graph(i, :) == 1);
  q(i, 2) = sum(A(i, :) == 1)/sum(graph(i, :) == 1);
  %q(i, 1) = 1 - p_t;
  %q(i, 2) = p_t;
end

display('just majority voting')
[row_max row_argmax] = max( q, [], 2 );
current_predictions = (row_argmax*2 - 3);
successful = sum(tasks == current_predictions)
total = n


pj = zeros(1, m);
for j = 1:m
  dj = find(graph(:, j) == 1);
  pj(j) = sum(q(dj + ((A(dj,j) + 3)/2-1)*n))/numel(dj);
end

% Log-likelihood function
% L(pj)

% 
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

%{
display('removing lowest reliability nodes')
[sm, sm_in] = getNElements(pj, 100);

A(:, sm_in) = 0;
graph(:, sm_in) = 0;
%}

% ti = ones(n, 1);
% for i = 1:n
%   di = find(graph(i, :) == 1);
%   ti(i) = sign(sum((2*pj(di) - 1).*A(i, di)));
% end

c = (row_argmax*2 - 3);
successful = sum(tasks == c)
total = n

a= find(tasks~=c);
qualn = abs(q(a, 1) - q(a, 2))';
b = find(tasks==c);
quale = abs(q(b, 1) - q(b, 2))';
qual_not = mean(qualn)
qual_eq = mean(quale)


qual = abs(q(:, 1) - q(:, 2));

mean(abs(p - pj'))



%{
%for run = 2:4
run = 1
%while sum(sum(graph)) < 2000*14
while 0
run = run +1;
fprintf('------- starting %d stage -----------\n', run);
l = 4;
r = 32;

[sorted idx] = sort(qual);
lower_idx = idx(1:ceil(n/2));


m_old = m;
m = ceil(numel(lower_idx)*l/r);


% reliablity of workers sampled from beta distribution
p_old = p;
p = 0.1 + 0.9*betarnd(alpha,beta,m,1);

% different graph generation
for k = 1:numel(lower_idx)
  i = lower_idx(k);
  indicesM = wrapS(k:k+l-1, m);
  graph(i, indicesM + m_old) = ones(l, 1);
  A(i, indicesM + m_old) = ((rand(l,1) <=  p(indicesM))*2 - 1)*tasks(i);
end

p = [p_old;p];


m = m + m_old;

display('just majority voting')
[row_max row_argmax] = max( q, [], 2 );
current_predictions = (row_argmax*2 - 3);
successful = sum(tasks == current_predictions)
total = n


pj = zeros(1, m);
for j = 1:m
  dj = find(graph(:, j) == 1);
  pj(j) = sum(q(dj + ((A(dj,j) + 3)/2-1)*n))/numel(dj);
end


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

end
%}

%{
[sorted idx] = sort(pj);
idx = idx(1:50);
graph(:, idx) = [];
A(:, idx) = [];
pj(idx) = [];
p(idx) = [];
m = m - numel(idx);
%}




%{
fprintf('------- starting third stage -----------\n');
%}