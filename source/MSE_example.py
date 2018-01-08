import sys
import numpy as np
import matplotlib.pyplot as plt
from plot import *
import textgrid as tg

def main():
    Fs = 100
    grid = tg.TextGrid("test")
    file_path = "../media/Convo_Sample.TextGrid"
    grid.read(file_path,Fs=Fs)

    # create a predicted yhat
    y = grid.FsArrayCombined
    yhat = y
    yhat = np.roll(yhat,int(Fs*0.2))
    
    # plot differences (if desired)
    changes,marks = tg.get_bin_changes(yhat,Fs)
    plot_bounds_lines(changes, marks, start=6, end=15)
    plot_bounds_fill(grid.FsTimeChangesCombined, grid.FsChangeMarksCombined, start=6, end=15)

    # print results
    print(validate_results(yhat,y))
    plt.show()
    
def validate_results(yhat, y):
    num_err = 0
    for i in range(len(yhat)):
        if yhat[i] != y[i]:
            num_err += 1
     
    training_accuracy = (len(yhat)-num_err)/len(yhat)*100
    print("%d Mistakes. Training Accuracy: %.2f%%"%(int(num_err),training_accuracy))
    return training_accuracy

if __name__ == '__main__':
  main()
