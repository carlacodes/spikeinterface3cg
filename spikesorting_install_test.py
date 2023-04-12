from dataclasses import dataclass
from pathlib import Path
import numpy as np
import os
import spikeinterface as si
import spikeinterface.sorters as ss
# import spikeinterface.extractors as se
# import spikeinterface.sorters as ss
# import spikeinterface.toolkit as st
# from spikeinterface.sorters import Kilosort2Sorter
# from spikeinterface.exporters import export_to_phy
# from spikeinterface import NumpyRecording, NumpySorting
# from spikeinterface import append_recordings, concatenate_recordings

# from probeinterface import generate_multi_columns_probe
#ss.Kilosort2Sorter.set_kilosort2_path('/home/zceccgr/Scratch/zceccgr/Kilosort-2.0') 
result = ss.installed_sorters()
print('installed sorters:')
print(result)
