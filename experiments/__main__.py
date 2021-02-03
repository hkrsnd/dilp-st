import sys
sys.path.append('datasets')
sys.path.append('datasets/append/')
sys.path.append('src/')
sys.path.append('experiments/')

from data_utils import DataUtils
from normal import normal_experiment
from noise import noise_experiment
from softor import softor_experiment
from step import step_experiment
import argparse


def main():
    print(sys.argv[1] + ' experiments runnning')
    parser = argparse.ArgumentParser(description='experiments')
    parser.add_argument('type', default='noise', type=str,
                        help='type of experiments [noise, step, softor, normal]')
    parser.add_argument('name', default='append', type=str,
                        help='name of the problem')
    parser.add_argument('lr', default=1e-2, type=float, help='learning rate')
    parser.add_argument('epoch', default=10000, type=int,
                        help='epoch in training')
    parser.add_argument('m', default=3, type=int,
                        help='the size of the solution')
    parser.add_argument('T', default=5, type=int, help='infer step')

    parser.add_argument('--noise_rate', default=0.00,
                        type=float, help='noise rate of training data')

    args = parser.parse_args()

    if args.type == 'noise':
        noise_experiment(args)
    elif args.type == 'normal':
        normal_experiment(args)
    elif args.type == 'step':
        step_experiment(args)
    elif args.type == 'softor':
        softor_experiment(args)


if __name__ == "__main__":
    main()
