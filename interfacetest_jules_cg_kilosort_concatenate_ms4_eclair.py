# Set logging before the rest as neo (and neo-based imports) needs to be imported after logging has been set
import logging
logging.basicConfig(
format='%(asctime)s %(levelname)-8s %(message)s',
level=logging.INFO,
datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('augusttest')
logger.setLevel(logging.DEBUG)

import datetime
import os
os.environ['NUMEXPR_MAX_THREADS'] = '36'
#doing 4x8 channel  map then removing the fourth row
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import os
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.toolkit as st
from spikeinterface.sorters import Kilosort2Sorter
from spikeinterface.exporters import export_to_phy, export_report

from spikeinterface import NumpyRecording, NumpySorting
from spikeinterface import append_recordings, concatenate_recordings

from probeinterface import generate_multi_columns_probe
logpath = Path('/home/zceccgr/Scratch/zceccgr/')
now = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')


fh = logging.FileHandler(logpath / f'multirec_sorting_logs_{now}.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)
logger.info('Starting')



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

        export_to_phy(self.we, '/home/zceccgr/Scratch/zceccgr/Electrophysiological_Data//F1902_Eclair//wpsoutput13',
                      compute_pc_features=False, compute_amplitudes=True, copy_binary=True)

    def save_ks_as_phy(self):
        self.we = si.WaveformExtractor.create(self.recording_preprocessed, self.sorting_KS, 'waveforms',
                                              remove_if_exists=True)

        self.we.set_params(ms_before=2., ms_after=2., max_spikes_per_unit=1000)
        self.we.run_extract_waveforms(n_jobs=3, chunk_size=30000)
        print(self.we)

        export_to_phy(self.we, '/home/zceccgr/Scratch/zceccgr/Electrophysiological_Data//F1902_Eclair//wpsoutput13',
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
    return recording_cmr


def run_ks2_cg(data, output_folder):
    params = {'projection_threshold' : [8, 3], 'detect_threshold': 5}
  
    data.sorting_KS = ss.run_kilosort2(recording=data,
                                       output_folder=output_folder, **params)

    print(data.sorting_KS)

def run_mountainsort4_cg(data, output_folder):
    #params = {'projection_threshold' : [8, 3], 'detect_threshold': 5}
    logger.info('running mountainsort4')

    data.sorting_mountainsort4 = ss.run_mountainsort4(recording=data,
                                       output_folder=output_folder)

    print(data.sorting_mountainsort4 )
    return data


def save_ks_as_phy_alone(data_test):
    # data_test.we = si.WaveformExtractor.create(self.recording_preprocessed, self.sorting_KS, 'waveforms',
    #                                       remove_if_exists=True)
    #
    # self.we.set_params(ms_before=2., ms_after=2., max_spikes_per_unit=1000)
    # self.we.run_extract_waveforms(n_jobs=3, chunk_size=30000)
    # print(self.we)

    export_to_phy(data_test, '/home/zceccgr/Scratch/zceccgr/Electrophysiological_Data//F1902_Eclair//wpsoutput13',
                  compute_pc_features=False, compute_amplitudes=True, copy_binary=True)

            
def save_as_phy_alone(data_test, rec):
    data_test.we = si.WaveformExtractor.create(rec, data_test.sorting_mountainsort4, 'waveforms',
                                            remove_if_exists=True)

    data_test.we.set_params(ms_before=2., ms_after=2., max_spikes_per_unit=1000)
    data_test.we.run_extract_waveforms(n_jobs=3, chunk_size=30000)
    print(data_test.we)
    export_to_phy(data_test.we, '/home/zceccgr/Scratch/zceccgr/Electrophysiological_Data//F1902_Eclair//wpsoutput29102022bb4bb5',
                        compute_pc_features=False, compute_amplitudes=True, copy_binary=True)
    export_report(data_test.we, '/home/zceccgr/Scratch/zceccgr/Electrophysiological_Data//F1902_Eclair//wpsoutput29102022bb4bb5//report//')



def main():
    #/home/zceccgr/Scratch/zceccgr/Electrophysiological_Data/F1902_Eclair
    datadir = Path('/home/zceccgr/Scratch/zceccgr/Electrophysiological_Data/F1902_Eclair')
    logger.info('at main function')
    dp = datadir / 'BlockNellie-162'
    store = ['BB_4', 'BB_5']
    recording_list = []

    ##If using kilosort, this spike sorter is going to call the latest version of MATLAB irrespective of what you use normally for kilosort,
    output_folder = Path('/home/zceccgr/Scratch/zceccgr/Electrophysiological_Data/F1902_Eclair/wpsoutputBB4BB5')
    print('hello, concatenating neural data blocks now')
    for i in range(31, 51): #range of intra trial roving stimuli for crumble
        print(i)
        block_ind = 'BlockNellie-' + str(i)
        print(block_ind)
        logger.info(f'variable: {block_ind}')

        dp2 = datadir / block_ind
        # if i == 6 or i==7 or i==2 or i==17:
        #     continue

        if os.path.isdir(dp2):
            print(i)
            data = TDTData(dp2, store)
            new_data = preprocess_data_cg(data)
        else:
            continue

        recording_list.append(new_data)
        #need to add condiitonal to check if streams are same length BB2 isequal BB3?

    rec = concatenate_recordings(recording_list)
    #only looking at the fourth row

    print(rec)
    logger.info(f'variable: {rec}')


    s = rec.get_num_samples(segment_index=0)
    print(f'segment {0} num_samples {s}')

    print('running mountainsort sorter now')
    

    data_test = run_mountainsort4_cg(rec, output_folder=output_folder)

    save_as_phy_alone(data_test, rec)


if __name__ == '__main__':
    print('running main test')
    main()
