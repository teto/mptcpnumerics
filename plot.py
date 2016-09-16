#!/usr/bin/env python3
import pandas as pd
import argparse

import logging
import matplotlib.pyplot as plt

delimiter = ","

def plot_buffers(csv_filename, out="output.png"):
     
    data = pd.read_csv(csv_filename, sep=delimiter,) 
    # d = data[ data["status"] != "Optimal"] 

    fig = plt.figure()

    axes = fig.gca()
    # print(d)
    # if not d.empty() :
    #     raise Exception("not everything optimal")

    data["objective"].hist(grid=True)

    fig.suptitle("Required buffer sizes", fontsize=12)

    axes.set_ylabel("Proportion")
    axes.set_xlabel("Inter DSN departure time")


    # filename = "output.png" # os.path.join(os.getcwd(), filename)
    # logger.info
    print("Saving into %s" % (out))
    fig.savefig(out)



def plot_cwnds(csv_filename, out="output.png"):
     
    data = pd.read_csv(csv_filename, sep=delimiter,) 
    # d = data[ data["status"] != "Optimal"] 

    fig = plt.figure()

    axes = fig.gca()
    # print(d)
    # if not d.empty() :
    #     raise Exception("not everything optimal")

    data["objective"].hist(grid=True)

    fig.suptitle("With constraints", fontsize=12)

    axes.set_ylabel("Proportion")
    axes.set_xlabel("Inter DSN departure time")


    # filename =  os.path.join(os.getcwd(), os.path.basename(filename))
    # logger.info
    print("Saving into %s" % (out))
    fig.savefig(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run tests")
    # parser.add_argument('')
    # plot_buffers()
    plot_cwnds("buffers.csv")
