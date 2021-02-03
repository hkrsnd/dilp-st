import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.use('Agg')

sns.set()
sns.set_style('darkgrid')
#sns.set_palette("winter", 8)
#plt.rcParams["font.size"] = 28
# plt.tight_layout()


def plot_loss(path, ys, title):
    pdf = PdfPages(path)
    plt.figure()
    plt.plot(ys, linewidth=0.5)
    plt.title(title, fontsize=20)
    pdf.savefig(bbox_inches='tight')
    pdf.close()
    plt.close()


def plot_loss_compare(path, ys_list, title, labels=['proposed', 'naive']):
    pdf = PdfPages(path)
    plt.figure()
    for i in range(len(ys_list)):
        plt.plot(ys_list[i], label=labels[i], linewidth=0.5)
    plt.title(title, fontsize=20)
    plt.legend()
    pdf.savefig(bbox_inches='tight')
    pdf.close()
    plt.close()


def plot_line_graph(path, xs, ys, xlabel, ylabel, title):
    pdf = PdfPages(path)
    plt.figure()
    #plt.plot(xs, ys)
    plt.plot(xs, ys, marker='o', markersize=14, color='blue', markerfacecolor='none',
             markeredgecolor='blue', markeredgewidth=4, label='proposed')
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    xticks = [xs[i] for i in range(len(xs)) if i % 2 == 0]
    plt.xticks(xticks, fontsize=14)
    plt.yticks(fontsize=18)
    plt.title(title, fontsize=20)
    # plt.legend(fontsize=18)
    pdf.savefig(bbox_inches='tight')
    pdf.close()
    plt.close()


def plot_line_graph_err(path, xs, ys, err, xlabel, ylabel, title):
    pdf = PdfPages(path)
    plt.figure()
    #plt.plot(xs, ys)
    plt.errorbar(xs, ys, yerr=err, marker='o', markersize=14, color='blue', markerfacecolor='none',
                 markeredgecolor='blue', markeredgewidth=4, label='proposed', capthick=1, capsize=10, lw=1)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    xticks = [xs[i] for i in range(len(xs)) if i % 2 == 0]
    plt.xticks(xticks, fontsize=16)
    plt.yticks(fontsize=18)
    plt.title(title, fontsize=20)
    # plt.legend(fontsize=18)
    pdf.savefig(bbox_inches='tight')
    pdf.close()
    plt.close()


def plot_line_graph_baseline(path, xs, ys, xlabel, ylabel, title, baseline):
    pdf = PdfPages(path)
    plt.figure()
    #plt.plot(xs, ys)
    plt.plot(xs, ys, marker='o', markersize=14, color='blue', markerfacecolor='none',
             markeredgecolor='blue', markeredgewidth=4, label='proposed')
    plt.plot(xs, baseline, color='black', linestyle='dashed', label='baseline')
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    xticks = [xs[i] for i in range(len(xs)) if i % 2 == 0]
    plt.xticks(xticks, fontsize=16)
    plt.yticks(fontsize=18)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=18)
    pdf.savefig(bbox_inches='tight')
    pdf.close()
    plt.close()


def plot_line_graph_baseline_err(path, xs, ys, err, xlabel, ylabel, title, baseline):
    pdf = PdfPages(path)
    plt.figure()
    #plt.plot(xs, ys)
    plt.errorbar(xs, ys, yerr=err, marker='o', markersize=14, color='blue', markerfacecolor='none',
                 markeredgecolor='blue', markeredgewidth=4, label='proposed', capthick=1, capsize=10, lw=1)
    plt.plot(xs, baseline, color='black', linestyle='dashed', label='baseline')
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    xticks = [xs[i] for i in range(len(xs)) if i % 2 == 0]
    plt.xticks(xticks, fontsize=16)
    plt.yticks(fontsize=18)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=18)
    pdf.savefig(bbox_inches='tight')
    pdf.close()
    plt.close()


def plot_line_graph_compare_err(path, xs, ys_list, err_list, xlabel, ylabel, title, labels, markers=['o', '^', 'x'], colors=['blue', 'red', 'green']):
    pdf = PdfPages(path)
    plt.figure()
    #plt.plot(xs, ys)
    for i in range(len(ys_list)):
        plt.errorbar(xs, ys_list[i], yerr=err_list[i], marker=markers[i], markersize=14, color=colors[i], markerfacecolor='none',
                     markeredgecolor=colors[i], markeredgewidth=4, label=labels[i], capthick=1, capsize=10, lw=1)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    #xticks = [xs[i] for i in range(len(xs)) if i % 2 == 0]
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(title, fontsize=20)
    plt.legend(fontsize=18)
    pdf.savefig(bbox_inches='tight')
    pdf.close()
    plt.close()


def __plot_line_graph(path, data, xlabel, ylabel):
    sns.lineplot(x=xlabel, y=ylabel, data=data)
    plt.savefig(path, dpi=300)
    plt.close()


def plot_line_graphs(path, xs, ys_list, labels, xlabel, ylabel, title):
    pdf = PdfPages(path)
    plt.figure()
    for i, ys in enumerate(ys_list):
        plt.plot(xs, ys, )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xs)
    plt.title(title)
    pdf.savefig()
    pdf.close()
    plt.close()


def plot_compare_line_graph(path, xs, ys_list, labels, xlabel, ylabel, title, markers=['o', '^', 'x'], colors=['blue', 'red', 'green']):
    # sns.set_style('darkgrid')
    #sns.set_palette("winter", 8)
    sns.set_style('whitegrid')
    pdf = PdfPages(path)
    plt.figure()
    for i, ys in enumerate(ys_list):
        plt.plot(xs, ys, marker=markers[i], markersize=14, label=labels[i], color=colors[i],
                 markerfacecolor='none',  markeredgecolor=colors[i], markeredgewidth=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xs)
    plt.title(title)
    plt.legend()
    pdf.savefig()
    pdf.close()
    plt.close()


def plot_compare_line_graph_xs_list(path, xs_list, ys_list, labels, xlabel, ylabel, title, markers=['o', '^', 'x'], colors=['blue', 'red', 'green']):
    sns.set_style('whitegrid')
    #sns.set_palette("winter", 8)
    pdf = PdfPages(path)
    plt.figure()
    for i in range(len(xs_list)):
        plt.plot(xs_list[i], ys_list[i], marker=markers[i], markersize=16, label=labels[i],
                 color=colors[i], markerfacecolor='none',  markeredgecolor=colors[i], markeredgewidth=4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.xticks(xs)
    plt.title(title)
    plt.legend()
    pdf.savefig()
    pdf.close()
    plt.close()


def plot_compare(path, results, result_ori, xs, ns):
    pdf = PdfPages(path)
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(xs)
    plt.title(title)
    pdf.savefig()
    pdf.close()
    plt.close()
