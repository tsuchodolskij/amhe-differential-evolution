import datetime
import os
import shutil

from algorithm import DifferentialEvolution
from noise import GaussianNoise
from noise import NoNoise
from noise import WhiteNoise
from matplotlib import pyplot as plt


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


def draw_plots(data, function_name, amps):
    plt.plot(list(range(0, len(data.best_points))), data.best_points)
    plt.title("Szum Biały, porównanie punktów najlepszych")
    plt.legend(["Ampl. " + str(amps[0]), "Ampl. " + str(amps[1]),
                "Ampl. " + str(amps[2]), "No noise"])
    plt.savefig('out/' + function_name + '_gauss_best')
    plt.clf()

    plt.plot(list(range(0, len(data.avg_points))), data.avg_points)
    plt.title("Szum Biały, porównanie średniej wartości punktów")
    plt.legend(["Ampl. " + str(amps[0]), "Ampl. " + str(amps[1]),
                "Ampl. " + str(amps[2]), "No noise"])
    plt.savefig('out/' + function_name + '_gauss_avg')
    plt.clf()

    plt.plot(list(range(0, len(data.variance))), data.variance)
    plt.title("Szum Biały, porównanie wariancji")
    plt.legend(["Ampl. " + str(amps[0]), "Ampl. " + str(amps[1]),
                "Ampl. " + str(amps[2]), "No noise"])
    plt.savefig('out/' + function_name + '_gauss_var')
    plt.clf()


if __name__ == '__main__':
    number_of_runs = 4
    val = 0
    clear_out_dir()

    amplitudes = [1, 10, 50, 0]
    func_name = 'rastrigin'

    for i in range(number_of_runs):
        print("\nIteration:", i + 1, "/", number_of_runs)
        start = datetime.datetime.now()
        de = DifferentialEvolution(num_iterations=150, dim=5, CR=0.4, F=0.48, population_size=40, print_status=False, func=func_name)
        noise = WhiteNoise(amplitude=amplitudes[i])
        de.noise = noise
        val += de.simulate()
        draw_plots(de, func_name, amplitudes)
