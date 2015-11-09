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

%No of batches
B = 10;

unif = rand(n,1);
% converting uniform distribution to 1 and -1 with probability p_t
tasks = (unif <= p_t)*2 - 1;

% reliablity of workers sampled from beta distribution
p = 0.1 + 0.9*betarnd(alpha,beta,m,1);

graph = zeros(n, m);
A = zeros(n, m);
%%{
%%generating a graph - adjacency matrix using configuration model
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
%%}

% different style of graph generation
% for i = 1:n
%   %indicesN = wrapS(i:i+l-1, n);
%   indicesM = wrapS(i:i+l-1, m);
%   graph(i, indicesM) = ones(l, 1);
%   A(i, indicesM) = ((rand(l,1) <=  p(indicesM))*2 - 1)*tasks(i);
% end

pj = 0.5*ones(m,1);

for batch = 2:B

    fprintf('*********batch:%d/%d*********', batch,B);
    
    m_batch = [(batch-1)*m/B + 1:1:batch*m/B]
    
    % q should mu_p(t|A). The columns correspond to {-1, 1}
    q = zeros(n, 2);
    % initialisation
    for i = 1:n
      q(i, 1) = sum(A(i, m_batch) == -1)/sum(graph(i, m_batch) == 1);
      q(i, 2) = sum(A(i, m_batch) == 1)/sum(graph(i, m_batch) == 1);
      %q(i, 1) = 1 - p_t;
      %q(i, 2) = p_t;
    end

    current_predictions = -2*(q(:,1) >= 0.5)+1;
    successful = sum(tasks == current_predictions);
    fprintf('Majority Voting: %d/%d\n', successful, n);

    for j = m_batch
      dj = find(graph(:, j) == 1);
      pj(j) = sum(q(dj + ((A(dj,j) + 3)/2-1)*n))/numel(dj);
    end

% Log-likelihood function
% L(pj)


    for k = 1:50
      % E-step
      p_vals = [1-pj pj];
      for i = 1:n
        di = find(graph(i, m_batch) == 1);
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
      for j = m_batch
        dj = find(graph(:, j) == 1);
        if(numel(dj) ~= 0)
            pj(j) = sum(q(dj + ((A(dj,j) + 3)/2-1)*n))/numel(dj);
        end
      end
    end

    display('running done')
    [row_max row_argmax] = max( q, [], 2 );
    c = (row_argmax*2 - 3);
    successful = sum(tasks == c)
    total = n

    [p(m_batch) pj(m_batch)]

    mean(abs(p(m_batch) - pj(m_batch)))

    for i=m_batch
       if((2*pj(i)-1)^2 <= 0.49)
           A(:,i) = 0;
           graph(:,i) = 0;
           pj(i) = 0.5;
       end
    end
    
    [p(m_batch) pj(m_batch)]
    
end