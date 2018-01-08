import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
import scipy.fftpack as fftpack
from scipy import signal
import sounddevice as sd
import matplotlib.animation as animation
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})


def track_audio(fs, start=0, end="end"):
    Y_MIN = -2
    Y_MAX = 2

    x = np.arange(start, end + 1, 0.01);

    def update_line(num, line):
        i = x[num]
        line.set_data([i, i], [Y_MIN, Y_MAX])
        return line,

    l, _ = plt.plot(start, -1, end, 1, linewidth=2, color='red')
    line_anim = animation.FuncAnimation(fig, update_line, len(x), fargs=(l,), interval=1 / fs, blit=True, repeat=False)
    plt.show()


def plot_spectrogram(x, fs, start=0, end="end"):
    if end is "end":
        Pxx, freqs, bins, im = plt.specgram(x[start * fs:-1],
                                            NFFT=512, Fs=fs, noverlap=100, cmap=plt.cm.gist_heat)
        end = max(bins)

    else:
        Pxx, freqs, bins, im = plt.specgram(x[start * fs:start * fs + end * fs],
                                            NFFT=512, Fs=fs, noverlap=100, cmap=plt.cm.gist_heat)
        if end > max(bins):
            end = max(bins)

    plt.ylim(0, max(freqs))
    plt.xlim([start, end])
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (Sec)")
    # plt.colorbar()


def play_audio(data, fs, start=0, end="end", blocking=True):
    if end is "end":
        sd.play(data[int(start * fs):-1], fs, blocking=blocking)
    else:
        sd.play(data[start * fs:int(start * fs + end * fs)], fs, blocking=blocking)


def plot_audio(x, fs, start=0, end="end"):
    if end is "end":
        xaxis = np.linspace(start, len(x) / fs, num=len(x[start * fs:]))
        plt.plot(xaxis, x[start * fs:] / max(x), linewidth=0.25)
        end = len(x) / fs

    else:
        if end > len(x) / fs:
            end = len(x) / fs
        xaxis = np.linspace(start, end, num=len(x[int(start * fs):int(end * fs)]))
        plt.plot(xaxis, x[int(start * fs):int(end * fs)] / max(x), color='#3030e0', linewidth=0.15)

    plt.ylim(-1.2, 1.2)
    plt.xlim([start, end])
    plt.ylabel("Magnitude")
    plt.xlabel("Time (Sec)")


def plot_bounds_lines(changes, marks, start=0, end="end"):
    x = changes
    if end is "end":
        last = x.shape[0] + 1
    else:
        last = np.argmax(x > end)

    data_range = range(np.argmin(x < start), last)

    if max(marks) > 2:
        plt.plot((-1, -1), (-1, -1), 'b-', linewidth=2, label="Speaker 1")
        plt.plot((-1, -1), (-1, -1), 'r-', linewidth=2, label="Speaker 2")
        plt.plot((-1, -1), (-1, -1), 'g-', linewidth=2, label="Both")
    else: 
        plt.plot((-1, -1), (-1, -1), 'b-', linewidth=2, label="Speech")

    plt.legend(loc=2, fancybox=True, framealpha=0.8, prop={'size': 9})

    for j, mark in enumerate(marks):
        if mark == 1:
            plt.plot((x[j], x[j + 1]), (-1.05, -1.05), 'b-', linewidth=2)
            plt.plot((x[j + 1], x[j + 1]), (-1.03, -1.07), 'k-', linewidth=2, zorder=10)

        elif mark == 2:
            plt.plot((x[j], x[j + 1]), (-1.05, -1.05), 'r-', linewidth=2)
            plt.plot((x[j + 1], x[j + 1]), (-1.03, -1.07), 'k-', linewidth=2, zorder=10)

        elif mark == 3:
            plt.plot((x[j], x[j + 1]), (-1.05, -1.05), 'g-', linewidth=2)
            plt.plot((x[j + 1], x[j + 1]), (-1.03, -1.07), 'k-', linewidth=2, zorder=10)

        else:
            pass

    plt.ylim(-1.2, 1.2)
    plt.xlim([start, end])
    plt.xlabel("Time (Sec)")


def plot_bounds_fill(changes, marks, start=0, end="end"):
    x = changes
    if end is "end":
        last = x.shape[0] + 1
    else:
        last = np.argmax(x >= end) + 1

    data_range = range(np.argmin(x < start), last)

    # create legend information
    # plt.axvspan(0, 0, alpha=0.5, color='b', label="\"N\" Human")
    # plt.axvspan(0, 0, alpha=0.5, color='r', label="\"S\" Human")
    # plt.axvspan(0, 0, alpha=0.5, color='g', label="\"S\" Human")
    plt.legend(loc=2, fancybox=True, framealpha=0.8, prop={'size': 9})

    for j, mark in enumerate(marks):
        if mark == 1:
            plt.axvspan(x[j], x[j + 1], alpha=0.2, color='b')

        elif mark == 2:
            plt.axvspan(x[j], x[j + 1], alpha=0.2, color='r')

        elif mark == 3:
            plt.axvspan(x[j], x[j + 1], alpha=0.2, color='g')

        else:
            pass


def plot_fft():
    N = 600  # sample points
    T = 1 / 800.0  # sample spacing
    x = np.linspace(0, N * T, N)
    y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
    yf = fftpack.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), N / 2)
    fig, ax = plt.subplots()
    ax.plot(xf, 2.0 / N * np.abs(yf[:N // 2]))


if __name__ == '__main__':
    main()
