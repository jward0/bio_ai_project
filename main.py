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
        pass

    def assess_performance(self, traffic_graph):
        pass

    def mutate(self):
        pass

    def crossover(self, other_routes):
        pass


def calculate_cost(traffic_graph, bus_routes):
    pass


def main():
    traffic_graph = TrafficGraph()
    traffic_graph.visualise_traffic()
    traffic_graph.visualise_demand()
    population = [BusRoutes() for _ in range(100)]


if __name__ == '__main__':
    main()
