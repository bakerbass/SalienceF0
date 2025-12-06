
import glob
import os

data_dir = r"c:\Users\ryanb\OneDrive\Documents\MyProjects\MyPythonProjects\AudioContentAnalysis\SalienceF0\Data\Training_Validation\MedleyDB-Pitch"
search_path = os.path.join(data_dir, "**", "*.wav")
wav_files = sorted(glob.glob(search_path, recursive=True))

print(f"Found {len(wav_files)} wav files.")
if len(wav_files) > 0:
    print(f"First file: {wav_files[0]}")
    
# Check pairing logic
found_pairs = 0
for wav_path in wav_files:
    basename = os.path.splitext(os.path.basename(wav_path))[0]
    parent_dir = os.path.dirname(wav_path) # .../audio
    if os.path.basename(parent_dir) == 'audio':
        grandparent = os.path.dirname(parent_dir)
        pitch_dir = os.path.join(grandparent, 'pitch')
        f0_path = os.path.join(pitch_dir, basename + ".csv")
        if os.path.exists(f0_path):
            found_pairs += 1

print(f"Found {found_pairs} matched pairs.")
