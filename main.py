import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import floyd_warshall


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

    def generate_random_cycle(self):
        start_vertex = (np.random.randint(0, 5), np.random.randint(0, 5))
        cycle = [start_vertex]
        banned_vertices = []

        cycle = self.select_next_cycle_step(cycle, banned_vertices)

        print(cycle)

        return cycle

    def select_next_cycle_step(self, cycle, banned_vertices):

        if len(cycle) > 1 and cycle[-1] == cycle[0]:
            return cycle

        (x, y) = cycle[-1]
        possible_moves = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        legal_moves = []

        for move in possible_moves:
            if move not in cycle[1:] and move not in banned_vertices and move[0] in range(5) and move[1] in range(5):
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
        plt.show()

    def assess_performance(self, traffic_graph):
        pass

    def mutate(self):
        pass

    def crossover(self, other_routes):
        pass


def calculate_cost(traffic_graph, bus_routes):
    pass


def main():
    # traffic_graph = TrafficGraph()
    # population = [BusRoutes() for _ in range(100)]
    routes = BusRoutes()
    routes.visualise()


if __name__ == '__main__':
    main()
