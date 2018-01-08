from unittest import TestCase
import textgrid as tg 


class TestTextGrid:
    def test_read_2_channel_grid(self):
        grid = tg.TextGrid("test")
        file_path = "../../media/Convo_Sample.TextGrid"
        grid.read(file_path,Fs=100)

        assert True

    def test_read_1_channel_grid(self):
        grid = tg.TextGrid("Sample")
        grid.read("../../media/Convo_Sample.TextGrid")

        Fs = 100
        # class 0: silence
        # class 1: speaker 1
        ch1 = grid.tiers[0].tier_to_array(Fs,grid.maxTime)

        assert True
