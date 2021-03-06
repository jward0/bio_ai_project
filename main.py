import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import floyd_warshall
import copy


class TrafficGraph:

    def __init__(self):
        # 12 time steps, 25 (5x5 grid) vertices
        # We want our default values for weights to be infinity (no connection)
        # And for desires to be 0 (no demand)
        self.traffic = np.ones(shape=(12, 25, 25)) * np.inf
        self.demand = np.zeros(shape=(12, 25, 25))

        # Initialise random weights and demands on edges between adjacent vertices
        for i in range(25):

            for j in range(25):
                self.demand[0][i][j] = np.random.randint(1, 11)

            if i % 5 != 0:
                self.traffic[0][i][i-1] = np.random.randint(1, 11)
            if i % 5 != 4:
                self.traffic[0][i][i+1] = np.random.randint(1, 11)
            if i > 4:
                self.traffic[0][i][i-5] = np.random.randint(1, 11)
            if i < 20:
                self.traffic[0][i][i+5] = np.random.randint(1, 11)

        # Populate all 12 time steps with random noise added to legal values, clipped to [0, 10]
        for i in range(11):
            for j in range(25):
                for k in range(25):
                    self.demand[i + 1][j][k] = np.clip(self.demand[i][j][k] + np.random.randint(-2, 3), 1, 10)
                    if self.traffic[i][j][k] != np.inf:
                        self.traffic[i+1][j][k] = np.clip(self.traffic[i][j][k] + np.random.randint(-2, 3), 1, 10)

    def visualise_single(self):
        coords = [(i % 5, int(i / 5)) for i in range(25)]

        fig, axs = plt.subplots(1, 2)

        for j in range(25):
            for k in range(25):
                if self.traffic[0][j][k] != np.inf:
                    axs[0].plot([coords[j][0], coords[k][0]], [coords[j][1], coords[k][1]],
                                   'r', linewidth=self.traffic[0][j][k])
                axs[1].plot([coords[j][0], coords[k][0]], [coords[j][1], coords[k][1]],
                               'g', alpha=0.1, linewidth=self.demand[0][j][k])

        # axs[0].gca().set_aspect('equal', adjustable='box')
        axs[0].title.set_text("Traffic (travel time between vertices)")
        # axs[1].gca().set_aspect('equal', adjustable='box')
        axs[1].title.set_text("Demand")

        plt.show()

    def visualise_traffic(self):
        coords = [(i % 5, int(i/5)) for i in range(25)]

        fig, axs = plt.subplots(3, 4)

        for i in range(12):
            for j in range(25):
                for k in range(25):
                    if self.traffic[i][j][k] != np.inf:
                        axs[int(i/4), i % 4].plot([coords[j][0], coords[k][0]], [coords[j][1], coords[k][1]],
                                                  'r', linewidth=self.traffic[i][j][k])

        plt.show()

    def visualise_demand(self):
        coords = [(i % 5, int(i/5)) for i in range(25)]

        fig, axs = plt.subplots(3, 4)

        for i in range(12):
            for j in range(25):
                for k in range(25):
                    axs[int(i/4), i % 4].plot([coords[j][0], coords[k][0]], [coords[j][1], coords[k][1]],
                                              'g', linewidth=self.demand[i][j][k])

        plt.show()


class BusRoutes:

    def __init__(self):
        # Generate 3 random cycles on the graph
        self.routes = [self.generate_random_cycle() for _ in range(3)]
        self.cost = np.inf

    def generate_random_cycle(self):
        start_vertex = (np.random.randint(0, 5), np.random.randint(0, 5))
        cycle = [start_vertex]
        banned_vertices = []

        cycle = self.select_next_cycle_step(cycle, banned_vertices)

        return cycle

    def select_next_cycle_step(self, cycle, banned_vertices):

        if len(cycle) > 1 and cycle[-1] == cycle[0]:
            return cycle

        (x, y) = cycle[-1]
        possible_moves = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        legal_moves = []

        for move in possible_moves:
            if not (move == cycle[0] and len(cycle) < 3):
                if move not in cycle[1:] and move not in banned_vertices and move in np.ndindex(5, 5):
                    legal_moves.append(move)

        if len(legal_moves) == 0:
            banned_vertices.append(cycle[-1])
            return self.select_next_cycle_step(cycle[:-1], banned_vertices)
        else:
            next_move = legal_moves[np.random.randint(0, len(legal_moves))]
            cycle.append(next_move)
            banned_vertices.append(next_move)
            return self.select_next_cycle_step(cycle, banned_vertices)

    def visualise(self):

        plt.plot([xy[0]+0.05 for xy in self.routes[0]], [xy[1]+0.05 for xy in self.routes[0]], 'r', alpha=0.4)

        plt.plot([xy[0] for xy in self.routes[1]], [xy[1] for xy in self.routes[1]], 'g', alpha=0.4)

        plt.plot([xy[0]-0.05 for xy in self.routes[2]], [xy[1]-0.05 for xy in self.routes[2]], 'b', alpha=0.4)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("Evolved bus routes")
        plt.show()

    def assess_performance(self, traffic_graph):
        pass

    def mutate_and_visualise(self):
        self.visualise()
        self.routes = [self.mutate(cycle) for cycle in self.routes]
        self.visualise()

    def mutate_all(self):
        self.routes = [self.mutate(cycle) for cycle in self.routes]

    @staticmethod
    def mutate(cycle):

        ndx = np.random.randint(0, len(cycle) - 2)

        if cycle[ndx + 1][0] != cycle[ndx - 1][0] and cycle[ndx + 1][1] != cycle[ndx - 1][1] and len(cycle) > 5:
            # Invert corner
            xy_0 = cycle[ndx - 1]
            xy_2 = cycle[ndx + 1]
            if cycle[ndx][0] == xy_0[0]:
                point = (xy_2[0], xy_0[1])
                if not (point in cycle and not(point == cycle[ndx-2] or point == cycle[ndx+2])):
                    cycle[ndx] = (xy_2[0], xy_0[1])

            else:
                point = (xy_0[0], xy_2[1])
                if not (point in cycle and not (point == cycle[ndx - 2] or point == cycle[ndx + 2])):
                    cycle[ndx] = (xy_0[0], xy_2[1])

            if ndx == 0:
                cycle[-1] = cycle[0]
            elif ndx == len(cycle)-1:
                cycle[0] = cycle[-1]
        else:
            # Move edge
            xy_0 = cycle[ndx]
            xy_1 = cycle[ndx+1]
            plus_flag = True
            neg_flag = True
            if xy_0[1] == xy_1[1]:
                # Move in +/- y
                # Check if vertices in +y are already in the cycle /and/ not at ndx-1, ndx+2
                if (xy_0[0], xy_0[1]+1) in cycle and (cycle[ndx-1] != (xy_0[0], xy_0[1]+1) or len(cycle) == 5):
                    plus_flag = False
                elif (xy_1[0], xy_1[1]+1) in cycle and (cycle[ndx+2] != (xy_1[0], xy_1[1]+1) or len(cycle) == 5):
                    plus_flag = False
                # Check if vertices in -y are already in the cycle /and/ not at ndx-1, ndx+2
                if (xy_0[0], xy_0[1]-1) in cycle and (cycle[ndx-1] != (xy_0[0], xy_0[1]-1) or len(cycle) == 5):
                    neg_flag = False
                elif (xy_1[0], xy_1[1]-1) in cycle and (cycle[ndx+2] != (xy_1[0], xy_1[1]-1) or len(cycle) == 5):
                    neg_flag = False

                # If both are valid mutations, choose one at random
                if plus_flag and neg_flag:
                    if np.random.randint(0, 2) == 0:
                        plus_flag = False
                    else:
                        neg_flag = False

                if plus_flag:
                    # Move +y
                    cycle.insert(ndx+1, (xy_1[0], xy_1[1]+1))
                    cycle.insert(ndx+1, (xy_0[0], xy_0[1]+1))
                elif neg_flag:
                    # Move -y
                    cycle.insert(ndx+1, (xy_1[0], xy_1[1]-1))
                    cycle.insert(ndx+1, (xy_0[0], xy_0[1]-1))
            else:
                # Move in +/- x
                # Check if vertices in +x are already in the cycle /and/ not at ndx-1, ndx+2
                if (xy_0[0]+1, xy_0[1]) in cycle and (cycle[ndx-1] != (xy_0[0]+1, xy_0[1]) or len(cycle) == 5):
                    plus_flag = False
                elif (xy_1[0]+1, xy_1[1]) in cycle and (cycle[ndx+2] != (xy_1[0]+1, xy_1[1]) or len(cycle) == 5):
                    plus_flag = False
                # Check if vertices in -x are already in the cycle /and/ not at ndx-1, ndx+2
                if (xy_0[0]-1, xy_0[1]) in cycle and (cycle[ndx-1] != (xy_0[0]-1, xy_0[1]) or len(cycle) == 5):
                    neg_flag = False
                elif (xy_1[0]-1, xy_1[1]) in cycle and (cycle[ndx+2] != (xy_1[0]-1, xy_1[1]) or len(cycle) == 5):
                    neg_flag = False

                # If both are valid mutations, choose one at random
                if plus_flag and neg_flag:
                    if np.random.randint(0, 2) == 0:
                        plus_flag = False
                    else:
                        neg_flag = False

                if plus_flag:
                    # Move +x
                    cycle.insert(ndx+1, (xy_1[0]+1, xy_1[1]))
                    cycle.insert(ndx+1, (xy_0[0]+1, xy_0[1]))
                elif neg_flag:
                    # Move -x
                    cycle.insert(ndx+1, (xy_1[0]-1, xy_1[1]))
                    cycle.insert(ndx+1, (xy_0[0]-1, xy_0[1]))

        # Trim redundant vertices

        unq, count = np.unique(cycle[1:], axis=0, return_counts=True)
        repeated_vertex = np.squeeze(unq[count > 1])

        if repeated_vertex.any():
            try:
                indices = np.where(np.all(np.array(cycle[1:]) == repeated_vertex, axis=1))
                for ndx in reversed(range(indices[0][0], indices[0][1])):
                    del cycle[ndx + 1]
            except np.AxisError:
                print("Failed")
                print(np.array(cycle[1:]))
                print(repeated_vertex)

        # Trim outside of bounds

        cycle = [vtx for vtx in cycle if 0 <= vtx[0] <= 4 and 0 <= vtx[1] <= 4]

        return cycle


def crossover(routes_1, routes_2):
    n = np.random.randint(0, 3)
    r1 = copy.deepcopy(routes_1)
    r2 = copy.deepcopy(routes_2)

    r1.routes[n], r2.routes[n] = r2.routes[n], r1.routes[n]

    return r1, r2


def calculate_cost(traffic_graph, bus_routes):
    # Construct graph from traffic graph and bus routes

    travel_time_graph = np.ones(shape=(12, 25, 25)) * np.inf
    shortest_distance_graph = np.ones(shape=(12, 25, 25)) * np.inf

    for route in bus_routes.routes:
        for ndx in range(len(route) - 1):
            j = route[ndx][0]*5 + route[ndx][1]
            k = route[ndx+1][0]*5 + route[ndx+1][1]
            for i in range(12):
                travel_time_graph[i][j][k] = traffic_graph.traffic[i][j][k]

    # Apply Floyd-Warshall to find shortest distances and apply to demand graph to find cost

    total_cost = 0

    for i in range(1):
        shortest_distance_graph[i] = floyd_warshall(travel_time_graph[i])
        total_cost += np.sum(np.multiply(shortest_distance_graph[i], traffic_graph.demand[i]))

    return total_cost


def iterate_ga(traffic_graph, original_population):

    # Calculate costs and cull invalid solutions

    for routes in original_population:
        routes.cost = calculate_cost(traffic_graph, routes)

    population = [routes for routes in original_population if routes.cost != np.inf]

    # Select top 50% of performers to pass forward
    population.sort(key=lambda x: x.cost)
    # print(f"Valid members: {len(population)}")
    # print([item.cost for item in population])
    print(f"Best generational performance: {population[0].cost}")
    mean_cost = sum(a.cost for a in population)/len(population)
    print(f"Mean generational performance: {mean_cost}")
    print(f"Valid member %: {len(population)/10}")
    population = population[:int(len(population)/2)]
    # print(f"Kept {len(population)} members with costs {population[0].cost} - {population[-1].cost}")
    # print([item.cost for item in population])

    new_population = []

    # Crossover- randomly until 1000 members
    for i in range(500):
        p_ndx = np.random.randint(0, len(population))
        q_ndx = np.random.randint(0, len(population))
        p = population[p_ndx]
        q = population[q_ndx]
        # print(f"foo {p_ndx}, {p.cost} {q_ndx}, {q.cost}")
        a, b = crossover(p, q)
        # a.cost = calculate_cost(traffic_graph, a)
        # b.cost = calculate_cost(traffic_graph, b)
        # print(f"bar {a.cost} {b.cost}")
        new_population.append(a)
        new_population.append(b)

    # Mutate
    for routes in new_population:
        if np.random.randint(0, 10) > 4:
            routes.mutate_all()

    return new_population


def main():
    traffic_graph = TrafficGraph()
    population = [BusRoutes() for _ in range(1000)]

    traffic_graph.visualise_single()
    best_cost = np.inf
    best_routes = BusRoutes()

    for i in range(30):
        print(f"Generation {i}")
        population = iterate_ga(traffic_graph, population)
        if population[0].cost < best_cost:
            best_cost = population[0].cost
            best_routes = population[0]

    best_routes.visualise()

    # routes = BusRoutes()
    # routes.mutate_and_visualise()
    # calculate_cost(traffic_graph, routes.routes)


if __name__ == '__main__':
    main()
