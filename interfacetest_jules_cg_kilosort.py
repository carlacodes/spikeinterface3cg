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

        export_to_phy(self.we, 'E:\\Electrophysiological_Data\\F1702_Zola_Nellie\\warpspikeinterface_output6',
                      compute_pc_features=False, compute_amplitudes=True, copy_binary=True)



def main():
    datadir = Path('E:\\Electrophysiological_Data\\F1702_Zola_Nellie')
    dp = datadir /'BlockNellie-162'
    store = ['BB_2', 'BB_3']
    # sorting_path = '\\home\\jules\\code\\WARPAutomatedSpikesorting\\output_spikesorting\\firings.npz'
    ##this spike sorter is going to call the latest version of MATLAB irrespective of what you actually use normally for kilosort, thus install parallel computing toolbox on that latest version of matlab
    output_folder = Path('E:\\Electrophysiological_Data\\F1702_Zola_Nellie\\warpspikeinterface_output5')

    data = TDTData(dp, store)
    data.preprocess_data()
    Kilosort2Sorter.set_kilosort2_path('D:\Scripts\Kilosort-2.0')
    #E:\Electrophysiological_Data\F1702_Zola_Nellie\warpspikeinterface_output3
    data.run_ks2(output_folder=output_folder)
    #data.run_kilosort2(output_folder=output_folder)
    # data.load_sorting(sorting_path)
    data.save_ks_as_phy()


if __name__ == '__main__':
    main()
