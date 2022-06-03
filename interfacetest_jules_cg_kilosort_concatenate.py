from dataclasses import dataclass
from pathlib import Path
import numpy as np
import os
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.toolkit as st
from spikeinterface.sorters import Kilosort2Sorter
from spikeinterface.exporters import export_to_phy
from spikeinterface import NumpyRecording, NumpySorting
from spikeinterface import append_recordings, concatenate_recordings

from probeinterface import generate_multi_columns_probe

result = si.sorters.installed_sorters()


@dataclass
class TDTData:
    dp: str
    store: list

    def load_tdtRec(self):
        self.recording = se.CustomTdtRecordingExtractor(self.dp, store=self.store)

    def preprocess_data(self):
        recording = se.CustomTdtRecordingExtractor(self.dp, store=self.store)

        probe = generate_multi_columns_probe(num_columns=8,
                                             num_contact_per_column=4,
                                             xpitch=350, ypitch=350,
                                             contact_shapes='circle')
        probe.create_auto_shape('rect')

        channel_indices = np.array([29, 31, 13, 15,
                                    25, 27, 9, 11,
                                    30, 32, 14, 16,
                                    26, 28, 10, 12,
                                    24, 22, 8, 6,
                                    20, 18, 4, 2,
                                    23, 21, 7, 5,
                                    19, 17, 3, 1])

        probe.set_device_channel_indices(channel_indices - 1)
        recording = recording.set_probe(probe)

        recording_f = st.bandpass_filter(recording, freq_min=300, freq_max=6000)
        print(recording_f)
        recording_cmr = st.common_reference(recording_f, reference='global', operator='median')
        print(recording_cmr)

        self.recording_preprocessed = recording_cmr
        return recording_cmr

        # this computes and saves the recording after applying the preprocessing chain
        # recording_preprocessed = recording_cmr.save(format='binary')

    def run_mountainsort(self, output_folder):
        self.sorting_MS = ss.run_mountainsort4(recording=self.recording_preprocessed,
                                               output_folder=output_folder)

        print(self.sorting_MS)

    def run_ks2(self, output_folder):
        self.sorting_KS = ss.run_kilosort2(recording=self.recording_preprocessed,
                                           output_folder=output_folder)

        print(self.sorting_KS)

    def load_sorting(self, sorting_path):
        self.sorting_MS = se.NpzSortingExtractor(sorting_path)

    def save_as_phy(self):
        self.we = si.WaveformExtractor.create(self.recording_preprocessed, self.sorting_MS, 'waveforms',
                                              remove_if_exists=True)

        self.we.set_params(ms_before=2., ms_after=2., max_spikes_per_unit=1000)
        self.we.run_extract_waveforms(n_jobs=3, chunk_size=30000)
        print(self.we)

        export_to_phy(self.we, 'E:\\Electrophysiological_Data\\F1702_Zola_Nellie\\warpspikeinterface_output2',
                      compute_pc_features=False, compute_amplitudes=True, copy_binary=True)

    def save_ks_as_phy(self):
        self.we = si.WaveformExtractor.create(self.recording_preprocessed, self.sorting_KS, 'waveforms',
                                              remove_if_exists=True)

        self.we.set_params(ms_before=2., ms_after=2., max_spikes_per_unit=1000)
        self.we.run_extract_waveforms(n_jobs=3, chunk_size=30000)
        print(self.we)

        export_to_phy(self.we, 'E:\\Electrophysiological_Data\\F1702_Zola_Nellie\\warpspikeinterface_output8',
                      compute_pc_features=False, compute_amplitudes=True, copy_binary=True)


def preprocess_data_cg(data):
    recording = se.CustomTdtRecordingExtractor(data.dp, store=data.store)

    probe = generate_multi_columns_probe(num_columns=8,
                                         num_contact_per_column=4,
                                         xpitch=350, ypitch=350,
                                         contact_shapes='circle')
    probe.create_auto_shape('rect')

    channel_indices = np.array([29, 31, 13, 15,
                                25, 27, 9, 11,
                                30, 32, 14, 16,
                                26, 28, 10, 12,
                                24, 22, 8, 6,
                                20, 18, 4, 2,
                                23, 21, 7, 5,
                                19, 17, 3, 1])

    probe.set_device_channel_indices(channel_indices - 1)
    recording = recording.set_probe(probe)

    recording_f = st.bandpass_filter(recording, freq_min=300, freq_max=6000)
    print(recording_f)
    recording_cmr = st.common_reference(recording_f, reference='global', operator='median')
    print(recording_cmr)

    # self.recording_preprocessed = recording_cmr
    return recording_cmr


def run_ks2_cg(data, output_folder):
    data.sorting_KS = ss.run_kilosort2(recording=data,
                                       output_folder=output_folder)

    print(data.sorting_KS)


def save_ks_as_phy_alone(data_test):
    # data_test.we = si.WaveformExtractor.create(self.recording_preprocessed, self.sorting_KS, 'waveforms',
    #                                       remove_if_exists=True)
    #
    # self.we.set_params(ms_before=2., ms_after=2., max_spikes_per_unit=1000)
    # self.we.run_extract_waveforms(n_jobs=3, chunk_size=30000)
    # print(self.we)

    export_to_phy(data_test, 'E:\\Electrophysiological_Data\\F1702_Zola_Nellie\\warpspikeinterface_output8',
                  compute_pc_features=False, compute_amplitudes=True, copy_binary=True)


def main():
    datadir = Path('E:\\Electrophysiological_Data\\F1702_Zola_Nellie')
    dp = datadir / 'BlockNellie-162'
    store = ['BB_2', 'BB_3']
    recording_list = []
    # sorting_path = '\\home\\jules\\code\\WARPAutomatedSpikesorting\\output_spikesorting\\firings.npz'
    ##this spike sorter is going to call the latest version of MATLAB irrespective of what you actually use normally for kilosort, thus install parallel computing toolbox on that latest version of matlab
    output_folder = Path('E:\\Electrophysiological_Data\\F1702_Zola_Nellie\\warpspikeinterface_output8')
    #if there are too many blocks concatenated you WILL run into a memory error depending on your GPU
    for i in range(115, 130):
        block_ind = 'BlockNellie-' + str(i)

        dp2 = datadir / block_ind
        if i == 131 or i==138 or i==146 or i==148 or i==150 or i==152 or i==169 or i==124 or i==140 or i==164:
            continue

        if os.path.isdir(dp2):
            print(i)
            data = TDTData(dp2, store)
            new_data = preprocess_data_cg(data)
        else:
            continue

        recording_list.append(new_data)
        #need to add condiitonal to check if streams are same length BB2 isequal BB3?

    rec = concatenate_recordings(recording_list)
    print(rec)
    s = rec.get_num_samples(segment_index=0)
    print(f'segment {0} num_samples {s}')

    print('running kilosort sorter now')
    Kilosort2Sorter.set_kilosort2_path('D:\Scripts\Kilosort-2.0')
    # E:\Electrophysiological_Data\F1702_Zola_Nellie\warpspikeinterface_output3
    data_test = run_ks2_cg(rec, output_folder=output_folder)
    # data.run_kilosort2(output_folder=output_folder)
    # data.load_sorting(sorting_path)
    # save_ks_as_phy_alone(data_test)


if __name__ == '__main__':
    main()
