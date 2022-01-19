import copy
import random

from population import Population
from objectives import Function


class DifferentialEvolution(object):
    def __init__(self, num_iterations=10, CR=0.4, F=0.48, dim=2, population_size=10, print_status=False, func=None):
        random.seed()
        self.print_status = print_status
        self.num_iterations = num_iterations
        self.iteration = 0
        self.CR = CR
        self.F = F
        self.population_size = population_size
        self.func = Function(func=func)
        self.population = Population(dim=dim, num_points=self.population_size, objective=self.func)

    def iterate(self):
        for ix in range(self.population.num_points):
            x = self.population.points[ix]
            [a, b, c] = random.sample(self.population.points, 3)
            while x == a or x == b or x == c:
                [a, b, c] = random.sample(self.population.points, 3)

            R = random.random() * x.dim
            y = copy.deepcopy(x)

            for iy in range(x.dim):
                ri = random.random()

                if ri < self.CR or iy == R:
                    y.coords[iy] = a.coords[iy] + self.F * (b.coords[iy] - c.coords[iy])

            y.evaluate_point()
            if y.z < x.z:
                self.population.points[ix] = y
        self.iteration += 1

    def simulate(self):
        pnt = self.get_best_point(self.population.points)
        print("Initial best value: " + str(pnt.z))
        while self.iteration < self.num_iterations:
            if self.print_status is True and self.iteration % 50 == 0:
                pnt = self.get_best_point(self.population.points)
                print(pnt.z, self.population.get_average_objective())
            self.iterate()

        pnt = self.get_best_point(self.population.points)
        print("Final best value: " + str(pnt.z))
        return pnt.z

    def get_best_point(self, points):
        best = sorted(points, key=lambda x: x.z)[0]
        return best
