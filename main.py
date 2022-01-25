import datetime
import os
import shutil
from copy import deepcopy

from algorithm import DifferentialEvolution
from noise import GaussianNoise
from noise import WhiteNoise
from matplotlib import pyplot as plt

from objectives import Function
from population import Population


def clear_out_dir():
    for filename in os.listdir('out'):
        file_path = os.path.join('out', filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def draw_plots(function_name, amps, best, avg, var, dims, n):
    splice1 = len(best[amps[0]])
    plt.plot(list(range(0, splice1)), best[amps[0]][:splice1])
    plt.plot(list(range(0, splice1)), best[amps[1]][:splice1])
    plt.plot(list(range(0, splice1)), best[amps[2]][:splice1])
    plt.plot(list(range(0, splice1)), best[amps[3]][:splice1])
    plt.title("Funkcja Rosenbrocka, szum biały, \nporównanie punktów najlepszych, " + str(dims) + " dim")
    plt.legend(["Ampl. " + str(amps[0]), "Ampl. " + str(amps[1]),
                "Ampl. " + str(amps[2]), "Brak szumu"])
    plt.savefig('out/' + function_name + '_' + str(dims) + 'dims_' + n + '_best')
    plt.clf()

    splice2 = len(avg[amps[0]])
    plt.plot(list(range(0, splice2)), avg[amps[0]][:splice2])
    plt.plot(list(range(0, splice2)), avg[amps[1]][:splice2])
    plt.plot(list(range(0, splice2)), avg[amps[2]][:splice2])
    plt.plot(list(range(0, splice2)), avg[amps[3]][:splice2])
    plt.title("Funkcja Rosenbrocka, szum biały, \nporównanie średniej wartości punktów, " + str(dims) + " dim")
    plt.legend(["Ampl. " + str(amps[0]), "Ampl. " + str(amps[1]),
                "Ampl. " + str(amps[2]), "Brak szumu"])
    plt.savefig('out/' + function_name + '_' + str(dims) + 'dims_' + n + '_avg')
    plt.clf()

    splice = len(var[amps[0]])
    plt.plot(list(range(0, splice)), var[amps[0]][:splice])
    plt.plot(list(range(0, splice)), var[amps[1]][:splice])
    plt.plot(list(range(0, splice)), var[amps[2]][:splice])
    plt.plot(list(range(0, splice)), var[amps[3]][:splice])
    plt.title("Funkcja Rosenbrocka, szum biały, \nporównanie wariancji, " + str(dims) + " dim")
    plt.legend(["Ampl. " + str(amps[0]), "Ampl. " + str(amps[1]),
                "Ampl. " + str(amps[2]), "Brak szumu"])
    plt.savefig('out/' + function_name + '_' + str(dims) + 'dims_' + n + '_var')
    plt.clf()


if __name__ == '__main__':
    number_of_runs = 4
    val = 0
    clear_out_dir()

    amplitudes = [1, 10, 50, 0]
    all_best_points = {amplitudes[0]: [], amplitudes[1]: [], amplitudes[2]: [], amplitudes[3]: []}
    all_avg_points = {amplitudes[0]: [], amplitudes[1]: [], amplitudes[2]: [], amplitudes[3]: []}
    all_variances = {amplitudes[0]: [], amplitudes[1]: [], amplitudes[2]: [], amplitudes[3]: []}

    pop_size = 100
    dim_size = 10
    func_name = 'rosenbrock'
    noise_name = 'white'
    population = Population(dim=dim_size, num_points=pop_size, objective=Function(func=func_name))

    for i in range(number_of_runs):
        print("\nIteration:", i + 1, "/", number_of_runs)
        start = datetime.datetime.now()
        de = DifferentialEvolution(num_iterations=40, CR=0.4, F=0.48, population_size=pop_size, print_status=False)
        de.population = deepcopy(population)
        noise = WhiteNoise(amplitude=amplitudes[i])
        # noise = GaussianNoise(amplitude=amplitudes[i])
        de.noise = noise
        val += de.simulate()
        all_best_points[amplitudes[i]] = de.best_points
        all_avg_points[amplitudes[i]] = de.avg_points
        all_variances[amplitudes[i]] = de.variance

    draw_plots(func_name, amplitudes, all_best_points, all_avg_points, all_variances, dim_size, noise_name)
