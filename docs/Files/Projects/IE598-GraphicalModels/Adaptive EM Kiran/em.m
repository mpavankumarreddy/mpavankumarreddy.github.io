%rng('default');
rng('shuffle');
%rng(2);

n = 2000;
p_t = 0.75;

wrapS = @(x, s) (1 + mod(x-1, s));

alpha = 6;
beta = 2;

l = 15;
r = 150;

m = n*l/r;

unif = rand(n,1);
% converting uniform distribution to 1 and -1 with probability p_t
tasks = (unif <= p_t)*2 - 1;

% reliablity of workers sampled from beta distribution
p = 0.1 + 0.9*betarnd(alpha,beta,m,1);


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

% different graph generation
for i = 1:n
  %indicesN = wrapS(i:i+l-1, n);
  indicesM = wrapS(i:i+l-1, m);
  graph(i, indicesM) = ones(l, 1);
  A(i, indicesM) = ((rand(l,1) <=  p(indicesM))*2 - 1)*tasks(i);
end


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

[p pj']

mean(abs(p - pj'))



% for i=1:m
%    if((2*pj(i)-1)^2 <= 0.25)
%        A(:,i) = 0;
%        pj(i) = 0;
%    end
% end
% 
% [p pj']

% display('just majority voting with reliable users')
% [row_max row_argmax] = max( q, [], 2 );
% current_predictions = (row_argmax*2 - 3);
% successful = sum(tasks == current_predictions)
% total = n
% 
% for k = 1:50
%   % E-step
%   p_vals = [1-pj;pj]';
%   for i = 1:n
%     di = find(graph(i, :) == 1);
%     ti = 1;
%     %prod_plus = prod(p_vals((A(i, di) == ti)*m + di));
%     prod_plus = p_t*prod(p_vals((A(i, di) == ti)*m + di));
%     ti = -1;
%     %prod_minus = prod(p_vals((A(i, di) == ti)*m + di));
%     prod_minus = (1-p_t)*prod(p_vals((A(i, di) == ti)*m + di));
%     q(i, 1) = prod_minus / (prod_minus + prod_plus);
%     q(i, 2) = prod_plus / (prod_minus + prod_plus);
%   end
% 
%   % M-step
%   for j = 1:m
%     dj = find(graph(:, j) == 1);
%     pj(j) = sum(q(dj + ((A(dj,j) + 3)/2-1)*n))/numel(dj);
%   end
% end
% 
% display('running done')
% [row_max row_argmax] = max( q, [], 2 );
% c = (row_argmax*2 - 3);
% successful = sum(tasks == c)
% total = n