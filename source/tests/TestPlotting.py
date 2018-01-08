import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from plot import *
import textgrid as tg 


def test_plot_1_channel():
    grid = tg.TextGrid("test")
    file_path = "../../media/Convo_Sample.TextGrid"
    grid.read(file_path,Fs=100)

    start = 6
    end = 20

    # note: channel 0 corresponds to grid 1
    Fs, x = wavfile.read("../../media/Convo Sample.wav")

    plt.subplot(2, 1, 1)
    plot_audio(x[:, 1], Fs, start=start, end=end)

    changes = grid.tiers[0].FsTimeChanges
    marks = grid.tiers[0].FsChangeMarks

    plot_bounds_lines(changes, marks, start=start, end=end)
    plot_bounds_fill(changes, marks, start=start, end=end)

    plt.subplot(2, 1, 2)
    plot_spectrogram(x[:, 1], Fs, start=start, end=end)

    # plt.ion()
    # play_audio(x[:, 1], Fs, start=start, end=end, blocking=False)
    plt.show()
    assert True

def test_plot_2_channels():
    grid = tg.TextGrid("test")
    file_path = "../../media/Convo_Sample.TextGrid"
    grid.read(file_path,Fs=100)

    start = 6
    end = 20

    # note: channel 0 corresponds to grid 1
    Fs, x = wavfile.read("../../media/Convo Sample.wav")

    plt.subplot(2, 1, 1)
    plot_audio(x[:, 1], Fs, start=start, end=end)

    changes = grid.FsTimeChangesCombined
    marks = grid.FsChangeMarksCombined

    plot_bounds_lines(changes, marks, start=start, end=end)
    plot_bounds_fill(changes, marks, start=start, end=end)

    plt.subplot(2, 1, 2)
    plot_spectrogram(x[:, 1], Fs, start=start, end=end)

    # plt.ion()
    # play_audio(x, Fs, start=start, end=end, blocking=False)
    plt.show()
    assert True
