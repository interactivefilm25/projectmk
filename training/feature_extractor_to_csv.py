import os
import subprocess
import shutil

# Paths
OPENSMILE_PATH = "C:/opensmile-3.0-win-x64/bin/SMILExtract.exe"
CONFIG_PATH = "C:/opensmile-3.0-win-x64/config/emobase/emobase.conf"

INPUT_FOLDER = "data/wav"
OUTPUT_FOLDER = "data/csv"

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Loop through .wav files and extract .csv features
for filename in os.listdir(INPUT_FOLDER):
    if filename.endswith(".wav"):
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_name = os.path.splitext(filename)[0].replace(" ", "_").replace("(", "").replace(")", "")
        output_path = os.path.join(OUTPUT_FOLDER, output_name + ".csv")

        try:
            print(f"🔧 Extracting features from: {filename}")
            subprocess.run([
                OPENSMILE_PATH,
                "-C", CONFIG_PATH,
                "-I", input_path,
                "-O", output_path
            ], check=True)
            print(f"✅ Saved to: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to process: {filename}")
            print(e)
df.to_csv('data/csv/features.csv', index=False)

print("\n🎉 Feature extraction completed.")