import random                   # @config
import numpy                    # @config
import networkx as nx           # @config
import itertools                # @config
from scipy.stats import beta    # @config
import matplotlib.pyplot as plt     # @config

import warnings

warnings.filterwarnings('error')


n = 2000    # @config
param_alpha = 6   # @config
param_beta = 2    # @config
max_l = 15  # @config

# n tasks with binary true labels i.e t = [t_1,...,t_n]
# t_i = {+1 w.p. 0.75, -1 else}
p_t = 0.75 # @config # probability that t_i is +1

# change seed to get different results
numpy.random.seed(1)

def gen_t_i():
    '''generates a random label with configured bernoulli distirbution'''
    if numpy.random.random() < p_t:
        return 1
    else:
        return -1

def gen_betarnd(m):
    '''
    returns a numpy array of m beta distributed random vars with
    configured alpha and beta
    '''
    return 0.1 + 0.9*numpy.random.beta(param_alpha, param_beta, (m,1))


class Task(object):
    '''A task in a crowdsourcing problem. Helper methods include -
    --------
    '''
    def __init__(self, task_id, label):
        self.task_id = task_id
        self.label = label


class Worker(object):
    '''Model for a worker in a crowdsourcing problem. Helper methods include -
    --------
    '''
    def __init__(self, worker_id, reliability):
        self.reliability = reliability
        self.worker_id = worker_id













gvkjs = None
gakjs = None
gadj_task_nodes = None
gpsi_k_j = None
ge = None


# bp update equations

def worker_to_task(G, tasks, workers):
    # one bp loop - later incorporate into a while loop
    #print 'worker to task'
    count = 0
    for e in G.edges_iter(data=True):
        print count,
        print ' ',
        count += 1
        # worker to task message updates

        worker_j = workers[e[2]['w_id']]

        adj_task_nodes = [out_edge['t_id'] for out_edge in G[worker_j].values()]
        adj_task_nodes.remove(e[2]['t_id'])

        vkjs = [G[tasks[i]][worker_j]['msg_t_w'] for i in adj_task_nodes]
        akjs = [G[tasks[i]][worker_j]['Aij'] for i in adj_task_nodes]

        for i in range(domain_size_p):
            p = domain_p[i]
            psi_k_j = [numpy.array([1-p, p])[::akj] for akj in akjs]
            #e[2]['msg_w_t'][i] = (numpy.prod([numpy.dot(vkjs[k], psi_k_j[k]) for k in range(len(adj_task_nodes))]))
            e[2]['msg_w_t'][i] = (numpy.prod([numpy.sum(numpy.multiply(numpy.multiply(vkjs[k], psi_k_j[k]), dist_t)) for k in range(len(adj_task_nodes))]))

        # normalising back to 1
        try:
            e[2]['msg_w_t'] /= sum(e[2]['msg_w_t'])
        except:
            print 'Oh noooo...'
            global ge
            global gvkjs
            global gakjs
            global gpsi_k_j
            global gadj_task_nodes
            ge = e
            gvkjs = vkjs
            gakjs = akjs
            gpsi_k_j = (psi_k_j)
            gadj_task_nodes = adj_task_nodes
            exit(-1)
    print ' '
    print 'worker to task is done'



def task_to_worker(G, tasks, workers):
    #print 'task to worker'
    count = 0
    for e in G.edges_iter(data=True):
        print count,
        print ' ',
        count += 1
        # task to worker message updates

        task_i = tasks[e[2]['t_id']]

        adj_worker_nodes = [out_edge['w_id'] for out_edge in G[task_i].values()]
        adj_worker_nodes.remove(e[2]['w_id'])

        vkis = [G[workers[k]][task_i]['msg_w_t'] for k in adj_worker_nodes]
        aiks = [G[workers[k]][task_i]['Aij'] for k in adj_worker_nodes]


        for i in range(domain_size_t):
            ti = domain_t[i]

            # If they are equal, 1 else -1
            e[2]['msg_t_w'][i] = numpy.prod([numpy.dot(t_vals[(int(ti == aiks[l]))], vkis[l]) for l in range(len(adj_worker_nodes))])

        # normalising back to 1
        e[2]['msg_t_w'] /= sum(e[2]['msg_t_w'])
    print ' '
    print 'task to worker is done!'


def estimate_ti(G, tasks, workers):
    t_hats = []
    for i in range(len(tasks)):
        t_hat_i = []

        task_i = tasks[i]

        adj_worker_nodes = [out_edge['w_id'] for out_edge in G[task_i].values()]

        vkis = [G[workers[k]][task_i]['msg_w_t'] for k in adj_worker_nodes]
        aiks = [G[workers[k]][task_i]['Aij'] for k in adj_worker_nodes]

        for i in range(domain_size_t):
            ti = domain_t[i]

            # If they are equal, 1 else -1
            t_hat_i.append(numpy.prod([numpy.dot(t_vals[(int(ti == aiks[l]))], vkis[l]) for l in range(len(adj_worker_nodes))]))

        sum_t = sum(t_hat_i)
        t_hats.append(t_hat_i)

    return t_hats


def estimate_pj(G, tasks, workers):
    w_hats = []
    for j in range(len(workers)):
        w_hat_j = []
        worker_j = workers[j]
        adj_task_nodes = [out_edge['t_id'] for out_edge in G[worker_j].values()]
        vkjs = [G[tasks[i]][worker_j]['msg_t_w'] for i in adj_task_nodes]
        akjs = [G[tasks[i]][worker_j]['Aij'] for i in adj_task_nodes]
        for i in range(domain_size_p):
            p = domain_p[i]
            psi_k_j = [numpy.array([1-p, p])[::akj] for akj in akjs]
            w_hat_j.append(numpy.prod([numpy.dot(vkjs[k], psi_k_j[k]) for k in range(len(adj_task_nodes))]) * dist_p[i])
        sum_w = sum(w_hat_j)
        w_hat_j = [temp/sum_w for temp in w_hat_j]
        w_hats.append(w_hat_j)
    return w_hats


def evaluate_ti_estimates(G, tasks, workers):
    # estimating task labels
    t_hats = estimate_ti(G, tasks, workers)
    t_est = numpy.array([t_h.index(max(t_h))*2 - 1 for t_h in t_hats])

    t_act = numpy.array([task.label for task in tasks])
    print "num correct - " + str(sum(t_est == t_act)) + " out of " + str(len(t_est))
    return (t_est, t_act)


def evaluate_wj_estimates(G, tasks, workers):
    # estimating worker reliability
    w_hats = estimate_pj(G, tasks, workers)
    w_est = numpy.array([domain_p[w_h.index(max(w_h))] for w_h in w_hats])

    w_act = numpy.array([worker.reliability for worker in workers])









# methods for generating a random l-r graph

def random_arr_permute(arr):
    '''
    Generates a random permutation of elements in the array arr using <whatever> algorithm. Swap x <-> arr[random(k-1)] assuming you are adding x at position k.
    '''
    for i in range(len(arr)):
        random_index = numpy.random.randint(0, i+1)     # to include i too
        arr[random_index], arr[i] = arr[i], arr[random_index]

def permuted_range_with_repetetitions(n, l):
    '''
    A custom method to give out a random permutation of numbers from 0 to n-1 each repeated l times'''
    nl = [i for i in range(n) for j in range(l)]
    random_arr_permute(nl)
    return nl

# generate random l-r graph
def configuration_model_connect(G, n, m, l, r):
    nl = permuted_range_with_repetetitions(n, l)
    mr = permuted_range_with_repetetitions(m, r)
    if n * l != m * r:
        print "Error"
        return
    for j in range(n*l):
        worker = workers[mr[j]]
        task = tasks[nl[j]]
        if worker not in G[task]:
            # generate responses based on Dawid-Skene model
            # worker.reliability seems to be an array yo!
            if numpy.random.random() < worker.reliability:
                aij = task.label
            else:
                aij = -task.label
            G.add_edge(worker, task, msg_t_w = None, msg_w_t = None, Aij = aij, t_id = nl[j], w_id = mr[j])

# generate some particular l-r graph
def graph_connect(G, n, m, l, r):
    for i in range(n):
        task = tasks[i]
        for j in range(l):
            worker_id = (i + j) % m
            worker = workers[worker_id]
            aij = -task.label
            if numpy.random.random() < worker.reliability:
                aij = task.label
            G.add_edge(worker, task, msg_t_w = None, msg_w_t = None, Aij = aij, t_id = i, w_id = worker_id)

tasks = [Task(i, gen_t_i()) for i in range(n)]

# A sample belief propagation on a random l-r
l = 15       # l - r random graph
r = 200      # l - r random graph

l_target = 6

m = int((n*l) / r)    # a free parameter

worker_reliabilities = gen_betarnd(m)   # Note that this is a m x 1 matrix
#workers = [Worker(reliability) for reliability in worker_reliabilities]
workers = [Worker(i, worker_reliabilities[i][0]) for i in range(m)]


G = nx.Graph()
G.add_nodes_from(tasks)
G.add_nodes_from(workers)

'''
print 'Generating random graph here'
configuration_model_connect(G, n, m, l, r)
print 'Done generating graph'
'''
print 'Generating one graph here'
graph_connect(G, n, m, l, r)
print 'Done generating graph'


# A simple quantization mechanism to get a discrete probability distribution over the beta distribution

domain_size_p = 5 # number of points in the discrete distribution

domain_p = numpy.concatenate((numpy.linspace(0, 0.5, int(domain_size_p*0.2))[1:], numpy.linspace(0.5, 1, int(0.8*domain_size_p))[1:-1]))
domain_size_p = len(domain_p)

dist_p = numpy.array([beta.pdf(i, param_alpha, param_beta) for i in domain_p])
dist_p = dist_p / sum(dist_p)

domain_t = [-1, 1]
domain_size_t = 2

dist_t = [1 - p_t, p_t]
#dist_t = [0.5, 0.5]

# a little bit optimisations
t_eq = numpy.multiply(domain_p, dist_p)
t_ne = numpy.multiply(1 - domain_p, dist_p)

# int(True) = 1, int(False) = 0
t_vals = [t_ne, t_eq]


# should we make messages log? Lets see, how big they are after first iteration
def init_graph(G):
    for e in G.edges_iter(data=True):
        # update msg_t_w to be lookup table of 2 value and initialise these messages to 1
        # WARNING: here [1, 1] corresponds to -1 and 1 values of ti. Didnot make it a dict for performance reasons
        e[2]['msg_t_w'] = numpy.array([1.0/domain_size_t]*domain_size_t)
        # update msg_w_t to be lookup table of domain_size_p and no initialisation required here
        # sorted keys order
        e[2]['msg_w_t'] = numpy.array([1.0/domain_size_p]*domain_size_p)

init_graph(G)

















print 'before bp'
evaluate_ti_estimates(G, tasks, workers);

w_act = numpy.array([worker.reliability for worker in workers])

def print_results(l):
    for i in range(l):
        print 'iteration - ' + str(i)
        worker_to_task(G, tasks, workers);
        task_to_worker(G, tasks, workers);
        evaluate_ti_estimates(G, tasks, workers)
        w_hats = estimate_pj(G, tasks, workers)
        w_est = numpy.array([domain_p[w_h.index(max(w_h))] for w_h in w_hats])

        print "mean error in reliability estimates - " + str(1.0/1000*sum(abs(w_est - w_act)**1))
        print "rms error in reliability estimates - " + str(numpy.sqrt(1.0/1000*sum(abs(w_est - w_act)**2)))

print_results(3)



'''
#exit(-1)
w_hats = estimate_pj(G, tasks, workers)
w_est = numpy.array([domain_p[w_h.index(max(w_h))] for w_h in w_hats])
for i in range(n):
    possible_connections = numpy.array([(i + j)%m for j in range(l_target)])
    w_est_connections = w_est[possible_connections]
    top_connections = possible_connections[w_est_connections.argsort()[-4:]]
    #print str(i) + ": - " + str(top_connections)
    task = tasks[i]
    for k in top_connections:
        worker = workers[k]
        if worker not in G[task]:
            aij = -task.label
            if numpy.random.random() < worker.reliability:
                aij = task.label
            G.add_edge(worker, task, msg_t_w = None, msg_w_t = None, Aij = aij, t_id = i, w_id = k)

init_graph(G)

print_results(3)

w_hats = estimate_pj(G, tasks, workers)
w_est = numpy.array([domain_p[w_h.index(max(w_h))] for w_h in w_hats])
for i in range(n):
    possible_connections = numpy.array([(i + j)%m for j in range(l_target)])
    w_est_connections = w_est[possible_connections]
    best_connections = possible_connections[w_est_connections.argsort()[-4:]]
    worst_connections = possible_connections[w_est_connections.argsort()[:2]]
    #print str(i) + ": - " + str(top_connections)
    task = tasks[i]
    for k in worst_connections:
        worker = workers[k]
        if worker in G[task]:
            G.remove_edge(task, worker)
    for k in best_connections:
        worker = workers[k]
        if worker not in G[task]:
            aij = -task.label
            if numpy.random.random() < worker.reliability:
                aij = task.label
            G.add_edge(worker, task, msg_t_w = None, msg_w_t = None, Aij = aij, t_id = i, w_id = k)

init_graph(G)
'''
