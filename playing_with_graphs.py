import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

class Node:

    def __init__(self, x, y):
        self.x = x
        self.y = y

class Route:

    def __init__(self, end_node: int, path: list, dist: float, heuristic: float):
        self.node = end_node
        self.path = path
        self.distance = dist
        self.heuristic = heuristic
        self.cost = self.distance + self.heuristic

    def __repr__(self):
        return f"""
        Route path: {' -> '.join(str(node) for node in self.path)}
        Total distance travelled: {self.distance}
        """

class RouteList:

    def __init__(self):
        self.routes = []

    def append(self, route):
        self.routes.append(route)

    def pop_shortest_route(self):
        cheapest_route = Route(0,[],np.inf, np.inf) # infinite route for starting with
        
        for route in self.routes:
            if route.cost < cheapest_route.cost:
                cheapest_route = route
        
        self.routes.remove(cheapest_route)
        return cheapest_route
    
    def __getitem__(self, idx):
        return self.routes[idx] 
    


class Graph:

    def __init__(self):
        self.nodes = []
        self.edges = np.empty((0,0))
    
    def add_node(self, node: Node):
        self.nodes.append(node) # add node to list
        self.edges = np.pad(self.edges, [(0,1),(0,1)], constant_values=np.inf)
        return True
    
    def connect(self, i, j, weight, both_ways=True):
        self.edges[i,j] = weight
        if both_ways:
            self.edges[j,i] = weight
        return True
    
    def get_euclidian_dist(self, i, j):
        i = self.nodes[i]
        j = self.nodes[j]

        dx = j.x - i.x
        dy = j.y - i.y

        dist = np.sqrt(dx ** 2 + dy ** 2)
        
        return dist
    
    def get_shortest_path(self, start: int, end: int, max_iters=1000):
        # TODO this breaks if the end node is not connected to the start node 
        # we should be able to gracefully catch the exception and pass an output that states that the end node is unreachable from the start node
        # implement A* search algorithm
        current_heuristic = self.get_euclidian_dist(start, end)
        current_route = Route(start, [start], dist=0, heuristic=current_heuristic)
        routes = RouteList()
        routes.append(current_route)

        iters = 0
        while end not in current_route.path:
            connections = self.edges[current_route.path[-1]] # get the possible connections which are determined by the edges array at the given row            
            if iters >= max_iters:
                #print("unable to find route in timesteps permitted")
                return False, current_route
                
            iters += 1
            
            for index, weight in enumerate(connections):
                if weight == np.inf:
                    continue # There is no path to this node (has infinite cost so not worth adding to list)
                else:
                    if len(current_route.path) > 1 and index in current_route.path: # if we've been to this node before then we should ignore it or we'll end up in a loop
                        continue # we are looking at the node we just came from

                    heuristic = self.get_euclidian_dist(index, end)
                    distance = current_route.distance + weight
                    path_to_node = current_route.path + [index]
                    route = Route(index, path_to_node, dist=distance, heuristic=heuristic)
                    routes.append(route)
            
            # now select cheapest route we have found and remove it from the RouteList. Using the end point of this route we do the above again
            try:
                current_route = routes.pop_shortest_route()
            except ValueError:
                #print("Unable to reach target node from starting node. These 2 nodes likely are not connected in the graph")
                return False, current_route
        
        return True, current_route

    
    def plot(self):
        # establish axes
        fig, ax = plt.subplots()

        # plot connections
        for i, column in enumerate(self.edges):
            for j, weight in enumerate(column):
                if weight == np.inf:
                    continue
                else:
                    x0, y0 = self.nodes[i].x, self.nodes[i].y
                    x1, y1 = self.nodes[j].x, self.nodes[j].y
                    ax.text((x0+x1) / 2, (y0+y1)/2, "{:.2f}".format(weight))
                    ax.plot([x0, x1], [y0, y1])
        
        for i, node in enumerate(self.nodes):
            x = node.x
            y = node.y
            
            ax.scatter(x,y, s=100)
            ax.text(x,y,i)

        plt.show()


def random_graph(size):
    my_graph = Graph()
        
    # generate some nodes
    nodes = []
    for i in range(size):
        x = np.random.rand() * 10
        y = np.random.rand() * 10
        nodes.append(Node(x,y))

    for node in nodes:
        my_graph.add_node(node)

    for index, node in enumerate(nodes):
        weight = np.random.rand() * 10
        endpoint = int(np.floor(np.random.rand() * len(nodes)))
        my_graph.connect(index, endpoint, weight)
    
    return my_graph

def test_random_graph():
        
        my_graph = random_graph(100)
        start = np.random.randint(0, len(my_graph.nodes))
        end = np.random.randint(0, len(my_graph.nodes))
        outcome, route = my_graph.get_shortest_path(start, end)
        return outcome


if __name__ == '__main__':

    successes = 0
    failures = 0

    for i in tqdm(range(10000)):
        try:
            outcome = test_random_graph()
            if outcome == True: 
                successes +=1
            else:
                failures += 1
        except KeyboardInterrupt:
            print("Testing terminated early")
            break
        except:
            failures +=1
            print("testing broke...")

    print(f'Successes: {successes} | Failures: {failures}')