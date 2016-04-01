from collections import defaultdict
import random, copy, pickle, itertools

class Inspection:
	def __init__(self, index, edge, mean, var, e_improve):
		self.index = index
		self.edge = edge
		self.mean = mean
		self.var = var
		self.e_improve = e_improve

class Dist:
	def __init__(self, gen_rv, mean, var, improve):
		self.gen_rv = gen_rv
		self.mean = mean
		self.var = var
		self.improve = improve
		self.realized = None

class Graph:
	def __init__(self):
		self.nodes = set()
		self.edges = defaultdict(set)
		self.distances = {}
	def add_node(self, value):
		self.nodes.add(value)
	def add_edge(self, from_node, to_node, distance):
		self.edges[from_node].add(to_node)
		self.edges[to_node].add(from_node)
		self.distances[(from_node, to_node)] = distance
		self.distances[(to_node, from_node)] = distance
	def get_edges(self):
		ret = set()
		for a, b in self.distances.keys():
			ret.add((min(a, b), max(a, b)))
		return ret

def gen_erdos_graph(n, p):
	g = Graph()
	for x in xrange(n):
		g.add_node(x)
	for i in xrange(n):
		for j in range(0, i):
			if random.random() < p:
				g.add_edge(i, j, unif(1))
	return g

def dijkstra(g, s):
	dist = {s: (0., 0.)}
	prev = {}
	nodes = copy.copy(g.nodes)
	while nodes:
		min_node = None
		for node in nodes:
			if node in dist:
				if min_node is None:
					min_node = node
				elif dist[node] < dist[min_node]:
					min_node = node
		if min_node is None:
			break
		nodes.remove(min_node)
		min_dist = dist[min_node]
		for neighbor in g.edges[min_node]:
			edge_dist = g.distances[(min_node, neighbor)]
			alt_dist = (min_dist[0] + edge_dist.mean, min_dist[1] + edge_dist.var)
			if neighbor not in dist or alt_dist < dist[neighbor]:
				dist[neighbor] = alt_dist
				prev[neighbor] = min_node
	return dist, prev

def floyd_warshall(g):
	n = len(g.nodes)
	dist = [[(float("inf"), float("inf")) for j in xrange(n)] for i in xrange(n)]
	for i in xrange(n):
		dist[i][i] = (0, 0)
	for (u, v) in g.get_edges():
		edge_dist = g.distances[(u, v)]
		edge_dist_tup = (edge_dist.mean, edge_dist.var)
		dist[u][v] = edge_dist_tup
		dist[v][u] = edge_dist_tup
	for k in xrange(n):
		for i in xrange(n):
			for j in xrange(n):
				alt_dist = (dist[i][k][0] + dist[k][j][0], dist[i][k][1] + dist[k][j][1])
				if dist[i][j] > alt_dist:
					dist[i][j] = alt_dist
	return dist

def path(prev, start, end):
	ret = set([])
	while end != start:
		ret.add((prev[end], end))
		end = prev[end]
	return ret

def greedy_inspect(g, start, end, candidates, all_pairs):
	greedy_edge = None
	max_improvement = -float("inf")
	baseline_dist = all_pairs[start][end][0]
	baseline_edges = path(dijkstra(g, start)[1], start, end)
	for (u, v) in candidates:
		improvement = 0
		edge_dist = g.distances[(u, v)]
		if (u, v) in baseline_edges:
			tmp = edge_dist.mean
			edge_dist.mean = float("inf")
			dijk_dists = dijkstra(g, start)[0]
			edge_dist.mean = tmp
			if end in dijk_dists:
				alt_dist = dijk_dists[end][0]
				if alt_dist < float("inf"):
					m = alt_dist - baseline_dist + edge_dist.mean 
					improvement = edge_dist.improve(True, m)
				else:
					improvement = 0
			else:
				improvement = 0
		else:
			rest_dist = min(all_pairs[start][u][0] + all_pairs[v][end][0], 
				all_pairs[start][v][0] + all_pairs[u][end][0])
			m = baseline_dist - rest_dist
			improvement = edge_dist.improve(False, m)
		if improvement > max_improvement:
			max_improvement = improvement
			greedy_edge = (u, v)
	return (greedy_edge, max_improvement)

def optimal_inspect(g, start, end, candidates, all_pairs, k):
	opt_edge = None
	min_e_dist = float("inf")
	edges = list(candidates)
	for edge in edges:
		candidates.remove(edge)
		dist = g.distances[edge]
		tmp_var = dist.var
		dist.var = 0
		tmp_mean = dist.mean
		dist.mean = 0
		e_dist_0 = optimal_inspect(g, start, end, candidates, all_pairs, k-1)[1] if k > 1 \
			else dijkstra(g, start)[0][end][0]
		dist.mean = 1
		e_dist_1 = optimal_inspect(g, start, end, candidates, all_pairs, k-1)[1] if k > 1 \
			else dijkstra(g, start)[0][end][0]
		candidates.add(edge)
		dist.var = tmp_var
		dist.mean = tmp_mean
		e_dist = (e_dist_0+e_dist_1)/2.
		if e_dist < min_e_dist:
			min_e_dist = e_dist
			opt_edge = edge
	return (opt_edge, min_e_dist)

def greedy_trial(g, start, end, k, budget = None):
	inspections = []
	candidates = g.get_edges()
	inspect_ct = 0
	greedy_edge = None
	budget = len(candidates) if budget == None else budget
	while True:
		all_pairs = floyd_warshall(g)
		best_dist = all_pairs[start][end]
		inspections.append(Inspection(inspect_ct, greedy_edge, best_dist[0], best_dist[1], None))
		if budget == 0:
			break
		if k == None:
			greedy_edge, _ = greedy_inspect(g, start, end, candidates, all_pairs) 
		else:
			greedy_edge, _ = optimal_inspect(g, start, end, candidates, all_pairs, 
				min(k, budget, len(candidates)))
		if greedy_edge == None:
			break
		else:
			candidates.remove(greedy_edge)
			dist = g.distances[greedy_edge]
			dist.mean = dist.realized
			dist.var = 0.0
			inspect_ct += 1
			budget -= 1
	return inspections

def random_trial(g, start, end):
	inspections = []
	all_edges = list(g.get_edges())
	random.shuffle(all_edges)
	best_dist = dijkstra(g, start)[0][end]
	inspections.append(Inspection(0, None, best_dist[0], best_dist[1], None))
	for i, v in enumerate(all_edges):
		dist = g.distances[v]
		dist.mean = dist.realized
		dist.var = 0.0
		best_dist = dijkstra(g, start)[0][end]
		inspections.append(Inspection(i+1, v, best_dist[0], best_dist[1], None))
	return inspections

def simulate(n, g, start, end, fname):
	with open(fname, 'w') as f:
		for i in xrange(n):
			r = copy.deepcopy(g)
			edges = r.get_edges()
			for edge in edges:
				dist = r.distances[edge]
				dist.realized = dist.gen_rv()
			random_copy = copy.deepcopy(r)
			random_inspections = random_trial(random_copy, start, end)
			greedy_copy = copy.deepcopy(r)
			greedy_inspections = greedy_trial(greedy_copy, start, end, None)
			to_print = map(lambda x: str(x.mean), random_inspections + greedy_inspections)
			f.write(",".join(to_print) + "\n")

def average(g, start, end, k, b, fname):
	with open(fname, 'w') as f:
		realizations = [list(x) for x in itertools.product([0,1], repeat=len(g.get_edges()))]
		edges = sorted(list(g.get_edges()))
		for real in realizations:
			r = copy.deepcopy(g)
			for i, edge in enumerate(edges):
				r.distances[edge].realized = real[i]
			optimal_inspections = greedy_trial(r, start, end, k, b)
			to_print = map(lambda x: str(x.mean), optimal_inspections)
			f.write(",".join(to_print) + "\n")

def unif(b):
	mu = b/2.0
	var = b*b/12.0
	def unif_improve(in_baseline, m):
		m = float(m)
		# m > mean when in_baseline
		if in_baseline:
			if m >= b:
				return 0.0
			else:
				lo = -0.5 * 0.5 * mu
				med = (m - mu)/b * (m - mu)/2.0
				hi = (b - m)/b * (m - mu)
				return -(lo + med + hi)
		else:
			change = 0
			if m > b:
				change = mu - m
			elif m > 0.0:
				change = (m/b) * (-m/2.0)
			else:
				change = 0.0
			return -change
	return Dist(lambda:random.uniform(0,b), mu, var, 
		lambda in_baseline, m: unif_improve(in_baseline, m))

def bern(a, b, pa):
	mu = pa*a + (1-pa)*b
	var = pa*a*a + (1-pa)*b*b - mu*mu
	def bern_improve(in_baseline, m):
		# m > mean when in_baseline
		if in_baseline:
			if m >= b:
				return 0.0
			else:
				lo = pa * (a - mu)
				hi = (1 - pa) * (m - mu)
				return -(lo + hi)
		else:
			change = 0
			if m > b:
				change = mu - m
			elif m > a:
				change = pa * (a - m)
			else:
				change = 0.0
			return -change
	return Dist(lambda: a if random.random() < pa else b, mu, var, 
		lambda in_baseline, m: bern_improve(in_baseline, m))

n3 = Graph()
for x in xrange(3):
	n3.add_node(x)

n3.add_edge(0,1,unif(1))
n3.add_edge(1,2,unif(1))
n3.add_edge(0,2,unif(2))

n4 = Graph()
for x in xrange(4):
	n4.add_node(x)

n4.add_edge(0,1,unif(1))
n4.add_edge(1,2,unif(1))
n4.add_edge(2,3,unif(1))

n4x1 = copy.deepcopy(n4)
n4x1.add_edge(0,3,unif(3))

n4x2 = copy.deepcopy(n4)
n4x2.add_edge(0,3,unif(3))
n4x2.add_edge(0,2,unif(2))

n4x3 = copy.deepcopy(n4)
n4x3.add_edge(0,3,unif(3))
n4x3.add_edge(1,3,unif(2))

n4x4 = copy.deepcopy(n4)
n4x4.add_edge(0,2,unif(2))
n4x4.add_edge(1,3,unif(2))

n4x5 = copy.deepcopy(n4)
n4x5.add_edge(0,3,unif(3))
n4x5.add_edge(0,2,unif(2))
n4x5.add_edge(1,3,unif(2))

n4x6 = Graph()
for x in xrange(4):
	n4x6.add_node(x)

n4x6.add_edge(0,1,unif(1))
n4x6.add_edge(1,3,unif(1))
n4x6.add_edge(0,2,unif(1))
n4x6.add_edge(2,3,unif(1))

n4x7 = copy.deepcopy(n4x6)
n4x7.add_edge(0,3,unif(2))

n5 = Graph()
for x in xrange(5):
	n5.add_node(x)

n5.add_edge(0,1,bern(0,1,0.5))
n5.add_edge(0,2,bern(0,1,0.5))
n5.add_edge(0,3,bern(0,1,0.5))
n5.add_edge(1,2,bern(0,1,0.5))
n5.add_edge(2,3,bern(0,1,0.5))
n5.add_edge(1,4,bern(0,1,0.5))
n5.add_edge(2,4,bern(0,1,0.5))
n5.add_edge(3,4,bern(0,1,0.5))

small_graphs = {
	"n3": n3,
	"n4x1": n4x1,
	"n4x2": n4x2,
	"n4x3": n4x3,
	"n4x4": n4x4,
	"n4x5": n4x5,
	"n4x6": n4x6,
	"n4x7": n4x7
}

### Simulations

# Simulate greedy and random policies on small graphs

for name, graph in small_graphs.items():
	end = len(graph.nodes) - 1
	simulate(10000, graph, 0, end, "compare/" + name + ".csv")

# Simulate lookahead policies on one small graph

for budget in range(1, 8):
	for lookahead in range(1, budget + 1):
		average(n5, 0, 4, lookahead, budget, "n5/lookahead" + str(lookahead) + "_budget" + str(budget) + ".csv")

average(n5, 0, 4, 1, 8, "n5/all_reveal.csv")

# Simulate greedy and random policies on 10 large graphs

nodes = 50
count = 0
while count < 10:
	erdos_graph = gen_erdos_graph(nodes, 0.05)
	all_pairs = floyd_warshall(erdos_graph)
	diameter = (-float("inf"), float("inf"))
	(max_i, max_j) = (-1, -1)
	for i in xrange(nodes):
		for j in xrange(nodes):
			if all_pairs[i][j][0] < float("inf") and all_pairs[i][j] > diameter:
				(max_i, max_j) = (i, j)
				diameter = all_pairs[i][j]
	reach = dijkstra(erdos_graph, max_i)[0]
	if diameter[0] > 5 and len(reach) > 40:
		count += 1
		g_name = "graph" + str(count)
		edges = erdos_graph.get_edges()
		with open("large_graphs/" + g_name + ".pickle", "w") as f:
			pickle.dump((erdos_graph.nodes, edges, max_i, max_j), f)
		simulate(1000, erdos_graph, max_i, max_j, "large_graphs/" + g_name + ".csv")

# Run more simulations of greedy and random policies on particular large graphs

for index in [1]:
	g_name = "graph" + str(index)
	with open("large_graphs/" + g_name + ".pickle") as f:
		(nodes, edges, start, end) = pickle.load(f)
		erdos_graph = Graph()
		erdos_graph.nodes = nodes
		for edge in edges:
			erdos_graph.add_edge(edge[0], edge[1], unif(1))
		simulate(1000, erdos_graph, start, end, "large_graphs/" + g_name + "_more.csv")
