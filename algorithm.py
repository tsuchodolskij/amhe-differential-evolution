import copy
import random
import numpy

from noise import Noise, NoNoise


class DifferentialEvolution(object):
    def __init__(self, num_iterations=10, CR=0.4, F=0.48, population_size=10, print_status=False, noise=None):
        random.seed()
        self.print_status = print_status
        self.num_iterations = num_iterations
        self.iteration = 0
        self.CR = CR
        self.F = F
        self.population_size = population_size
        self.noise = noise if isinstance(noise, Noise) else NoNoise()
        self.population = None
        self.best_points = []
        self.worst_points = []
        self.avg_points = []
        self.variance = []

    def iterate(self):
        for ix in range(self.population.num_points):
            x = self.population.points[ix]
            [a, b, c] = random.sample(self.population.points, 3)
            while x == a or x == b or x == c:
                [a, b, c] = random.sample(self.population.points, 3)

            y = copy.deepcopy(x)

            for iy in range(x.dim):
                ri = random.random()

                if ri < self.CR:
                    y.coords[iy] = a.coords[iy] + self.F * (b.coords[iy] - c.coords[iy])

            noise_value = self.noise.evaluate(y.coords)
            y.evaluate_point(noise_value)
            if y.z_noised < x.z_noised:
                self.population.points[ix] = y
        self.iteration += 1

    def simulate(self):
        while self.iteration < self.num_iterations:
            self.iterate()
            self.best_points.append(self.population.get_best_point().z)
            self.worst_points.append(self.population.get_worst_point().z)
            self.avg_points.append(self.population.get_average_objective())
            self.variance.append(numpy.var(list(map(lambda x: x.z, self.population.points))))

        pnt = self.population.get_best_point()

        return pnt.z
