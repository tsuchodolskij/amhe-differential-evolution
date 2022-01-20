import datetime
import os
import shutil

from algorithm import DifferentialEvolution


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


if __name__ == '__main__':
    number_of_runs = 5
    val = 0
    clear_out_dir()

    for i in range(number_of_runs):
        print("\nIteration:", i + 1, "/", number_of_runs)
        start = datetime.datetime.now()
        de = DifferentialEvolution(num_iterations=500, dim=50, CR=0.4, F=0.48, population_size=75, print_status=False, func='rastrigin')
        val += de.simulate()
        print("\nTime taken:", datetime.datetime.now() - start)
