%rng('default');
rng('shuffle');
%rng(2);

l_0 = 15;
n = 2000;
p_t = 0.75;
alpha = 6;
beta = 2;

parts = 4;

budget = l_0*n;
worker_frac = 1;
r = n/worker_frac;

tasks = (rand(n,1) <= p_t)*2 - 1;

p = zeros(0,1);
A = zeros(n,0);
pj = zeros(0,1);
% q should mu_p(t|A). The columns correspond to {-1, 1}
q = zeros(n,1);

task_index = 1:n;
n_left = n;
bottom_qual_partition = 2;
bottom_degree_partition = 100;
while(budget/n >= 1)

    fprintf('***********Budget = %d/%d***********\n', budget, l_0*n)

    l = floor(budget/n_left/parts);
    if(l == 0)
       l = 1; 
    end

    r = max(2,floor(n_left/worker_frac));
    m = round(n_left*l/r);

    p = [p; 0.1 + 0.9*betarnd(alpha,beta,m,1)];
    m_batch = (size(p,1) - m)+1:size(p,1);
    
    % generating a graph - adjacency matrix using configuration model
      
    left_half = reshape(repmat(task_index,l,1),n_left*l,[]);
    right_half = reshape(repmat(m_batch,r,1),m*r,[]);
    index = randperm(n_left*l);
    right_half = right_half(index);
    A = [A zeros(n, m)];
    for i = 1:(n_left*l)
      A(left_half(i), right_half(i)) = -tasks(left_half(i));
      if (rand() <= p(right_half(i)))
        A(left_half(i), right_half(i)) = tasks(left_half(i));
      end
    end

    if(budget == l_0*n)
        for i = 1:n
          q(i) = sum(A(i, :) == -1)/sum(A(i, :) ~= 0);
        end
        current_predictions = -2*(q >= 0.5)+1;
        successful = sum(tasks == current_predictions);
        %fprintf('Majority Voting: %d/%d\n', successful, n);
    end
    
    pj = [pj; zeros(m, 1)];
    for j = m_batch
      dj = find(A(:, j) ~= 0);
      pj(j) = sum(q(dj).*(A(dj,j)==-1) + (1-q(dj)).*(A(dj,j)==1))/numel(dj);
    end

    m_total = size(p,1);
    for k = 1:80
      %%% E-step
      p_vals = [1-pj;pj]';
      for i = 1:n
        di = find(A(i, :) ~= 0);
        prod_plus = p_t*prod(p_vals((A(i, di) == 1)*m_total + di));
        prod_minus = (1-p_t)*prod(p_vals((A(i, di) == -1)*m_total + di));
        q(i) = prod_minus / (prod_minus + prod_plus);
      end

      %%% M-step
      for j = 1:m_total
        dj = find(A(:, j) ~= 0);
        pj(j) = sum(q(dj).*(A(dj,j)==-1) + (1-q(dj)).*(A(dj,j)==1))/numel(dj);
      end
    end

    current_predictions = -2*(q >= 0.5)+1;
    successful = sum(tasks == current_predictions);
    fprintf('EM with budget=%d: %d/%d\n', l_0*n - budget + sum(sum(A(:,(size(pj,1) - m)+1:size(pj,1)) ~= 0)), successful, n);

    budget = budget - sum(sum(A(:,(size(pj,1) - m)+1:size(pj,1)) ~= 0));
    fprintf('Budget left=%d\n', budget);
    
    n_qual = ceil(n/bottom_qual_partition);
    
    
    task_qual_1 = find(abs(2*q - 1) < 1);
    [~, task_qual_index] = sort(abs(2*q - 1), 'ascend');
    
    %task_qual_index  = intersect(task_qual_index(1:n_qual), task_qual_1);
    %n_qual = numel(task_qual_index);
    
    n_degree = ceil(n/bottom_degree_partition);
    
    [~, task_degree_index] = sort(sum(abs(A),2), 'ascend');
    task_index = [task_qual_index(1:n_qual); task_degree_index(1:n_degree)];
    n_left = n_qual + n_degree;
end


error = (n - successful)/n; % uncomment

qual_task = abs(2*q - 1);
fprintf('mean of correct tasks: %d\n',sum(qual_task.*(tasks == current_predictions))/sum(tasks == current_predictions))
fprintf('mean of wrong tasks: %d\n',sum(qual_task.*(tasks ~= current_predictions))/sum(tasks ~= current_predictions))   
fprintf('no. of correct tasks: %d\n',sum(tasks == current_predictions))
fprintf('no. of wrong tasks: %d\n',sum(tasks ~= current_predictions))
q(tasks ~= current_predictions);
q(tasks == current_predictions);
numel(qual_task(qual_task < 0.9999));

degree_task = sum(abs(A),2);
degree_task(tasks ~= current_predictions);
degree_task(tasks == current_predictions);
sort(degree_task);