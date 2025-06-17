import opensmile
import numpy as np
import pandas as pd

sampling_rate = 16000
duration = 1.0
frequency = 440.0
time = np.linspace(0., duration, int(sampling_rate * duration), endpoint=False)
signal = np.sin(2 * np.pi * frequency * time).astype(np.float32)

print(f"Generated a NumPy array with:")
print(f"  Shape: {signal.shape}")
print(f"  Data type: {signal.dtype}\n")

smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

print(f"Initialized opensmile with feature set: {smile.feature_set.name}\n")
print("Processing the numpy array...")
features = smile.process_signal(
    signal=signal,
    sampling_rate=sampling_rate
)
print("Processing complete!\n")

print("Extracted Features (as a pandas DataFrame):")
print("Extracted Features, Pandas Dataframe:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(features)