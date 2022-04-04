import numpy as np
from scipy.sparse.csgraph import floyd_warshall


class TrafficGraph:

    def __init__(self):
        self.weights = np.empty(shape=(12, 25, 25))
        self.desires = np.empty(shape=(12, 25, 25))

        # We want our default values for weights to be infinity (no connection)
        # And for desires to be 0 (no demand)
        self.weights[0] = np.ones(shape=(25, 25)) * np.inf
        self.desires[0] = np.zeros(shape=(25, 25))

        # Initialise random weights and demands on edges between adjacent vertices
        for i in range(25):
            if i % 5 != 0:
                self.weights[i][i-1] = np.randint(5, 16)
                self.desires[i][i-1] = np.randint(5, 16)
            if i % 5 != 4:
                self.weights[i][i+1] = np.randint(5, 16)
                self.desires[i][i+1] = np.randint(5, 16)
            if i > 4:
                self.weights[i][i-5] = np.randint(5, 16)
                self.desires[i][i-5] = np.randint(5, 16)
            if i < 20:
                self.weights[i][i+5] = np.randint(5, 16)
                self.desires[i][i+5] = np.randint(5, 16)

        for i in range(11):
            self.weights[i+1] = self.weights[i]
            self.desires[i+1] = self.desires[i]


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
    population = [BusRoutes() for _ in range(100)]


if __name__ == '__main__':
    main()
