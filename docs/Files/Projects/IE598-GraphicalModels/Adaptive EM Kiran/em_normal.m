display('Initialization...') 
%rng('default');
rng('shuffle');
%rng(2);

n = 2000;
p_t = 0.75;

alpha = 6;
beta = 2;

tasks = (rand(n,1) <= p_t)*2 - 1;

l = 15;
r = n/2000;

m = round(n*l/r);
   
% reliablity of workers sampled from beta distribution
p = 0.1 + 0.9*betarnd(alpha,beta,m,1);

%%% generating a graph 
%Method 1 - adjacency matrix using configuration model
left_half = reshape(repmat(1:n,l,1),n*l,[]);
right_half = reshape(repmat((size(p,1) - m) + 1 : size(p,1),r,1),m*r,[]);
index = randperm(n*l);
right_half = right_half(index);
gaph = zeros(n, m);
A = zeros(n, m);
for i = 1:(n*l)
  graph(left_half(i), right_half(i)) = 1;
  A(left_half(i), right_half(i)) = -tasks(left_half(i));
  if (rand() <= p(right_half(i)))
    A(left_half(i), right_half(i)) = tasks(left_half(i));
  end
end
%Method II - Sewoong random edge pesent or not
% gaph = zeros(n, m);
% A = zeros(n, m);
% graph = ceil( rand(n,m)-1+(l/m) );
% for i = 1:n
%   di = find(graph(i, :) == 1);
%   A(i, di) = ((rand(numel(di), 1) <= p(di))*2 - 1) * tasks(i);
% end

q = zeros(n,1);
for i = 1:n
  q(i) = sum(A(i, :) == -1)/sum(A(i, :) ~= 0);
end

current_predictions = -2*(q > 0.5)+1;
successful = sum(tasks == current_predictions);
fprintf('Majority Voting: %d/%d\n', successful, n);

pj = zeros(m, 1);
for j = 1:m
  dj = find(A(:, j) ~= 0);
  pj(j) = sum(q(dj).*(A(dj,j)==-1) + (1-q(dj)).*(A(dj,j)==1))/numel(dj);
end

display('Initialization Done')
display('EM Algorithm started')

for k = 1:50
  %%% E-step
  p_vals = [1-pj;pj];
  for i = 1:n
    di = find(A(i, :) ~= 0);
    prod_plus = p_t*prod(p_vals((A(i, di) == 1)*m + di));
    prod_minus = (1-p_t)*prod(p_vals((A(i, di) == -1)*m + di));
    q(i) = prod_minus / (prod_minus + prod_plus);
  end

  %%% M-step
  for j = 1:m
    dj = find(A(:, j) ~= 0);
    pj(j) = sum(q(dj).*(A(dj,j)==-1) + (1-q(dj)).*(A(dj,j)==1))/numel(dj);
  end
end

display('EM Algorithm finshed')

current_predictions = -2*(q > 0.5)+1;
successful = sum(tasks == current_predictions);
fprintf('EM with budget=%d: %d/%d\n', sum(sum(A(:,(size(pj,1) - m)+1:size(pj,1)) ~= 0)), successful, n);

qual_task = abs(2*q - 1);
fprintf('mean of correct tasks: %d\n',sum(qual_task.*(tasks == current_predictions))/sum(tasks == current_predictions))
fprintf('mean of wrong tasks: %d\n',sum(qual_task.*(tasks ~= current_predictions))/sum(tasks ~= current_predictions))   
fprintf('no. of correct tasks: %d\n',sum(tasks == current_predictions))
fprintf('no. of wrong tasks: %d\n',sum(tasks ~= current_predictions))
q(tasks ~= current_predictions);
q(tasks ~= current_predictions);