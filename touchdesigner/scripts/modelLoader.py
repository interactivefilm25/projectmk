from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import librosa
import torch
import numpy as np
import time

print("Model Loader: Initializing and loading models...")
start_time = time.time()

# These lines will run ONCE when this script is first compiled by TouchDesigner.
# This may cause a one-time freeze on startup, which is expected.

#model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
model_id = "superb/hubert-large-superb-er"
#model_id = "superb/wav2vec2-base-superb-er"

model = AutoModelForAudioClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = model.config.id2label
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

if device.type == 'cuda':
	model = model.half()
	print("Using half precision (FP16) on GPU")

print(f"Model loaded to {device} in {time.time() - start_time:.2f} seconds.")

def remap_labels(predicted_label):
    if predicted_label in ["ang", "angry", "fearful"]:
        return "anger_fear"
    elif predicted_label in ["sad", "sadness"]:
        return "sadness"
    elif predicted_label in ["hap", "happy"]:
        return "joy_excited"
    elif predicted_label in ["neu", "neutral", "calm"]:
        return "calm_content"
    elif predicted_label == "surprised":
        return "curious_reflective"
    else:
        return 'other'

def preprocess_audio_file(audio_path, feature_extractor):
    """Preprocesses an audio file from disk."""
    audio_array, sampling_rate = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    
    inputs = feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    return inputs
    
def preprocess_numpy_audio(audio_array, sampling_rate, feature_extractor):
    """Preprocesses a NumPy audio array directly from a TouchDesigner CHOP."""
    if audio_array.ndim > 1:
        audio_array = audio_array.flatten()
        
    inputs = feature_extractor(
        audio_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    return inputs

def predict_emotion_from_file(audio_path):
    """Core function for the FILE-BASED background worker"""
    global model, feature_extractor, id2label, device
    try:
        inputs = preprocess_audio_file(audio_path, feature_extractor)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        if model.dtype == torch.float16:
            inputs = {key: val.half() for key, val in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        predicted_label = id2label[predicted_id]
        
        return remap_labels(predicted_label)
            
    except Exception as e:
        print(f"Error during prediction for {audio_path}: {e}")
        return f"ERROR: {e}"
        
def predict_emotion_from_numpy(audio_array, sampling_rate):
    """Core function for the faster, recommended NUMPY-BASED (CHOP) setup."""
    global model, feature_extractor, id2label, device
    try:
        print("Predict Func]: Starting preprocessing...")
        inputs = preprocess_numpy_audio(audio_array, sampling_rate, feature_extractor)
        inputs = {key: val.to(device) for key, val in inputs.items()}

        if model.dtype == torch.float16:
            inputs = {key: val.half() for key, val in inputs.items()}

        with torch.no_grad():
            print("[Predict Func]: >>> Entering model inference.")
            outputs = model(**inputs)
            print("[Predict Func]: <<< Exited model inference.")

        logits = outputs.logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        predicted_label = id2label[predicted_id]
        
        print(predicted_id)
        
        return predicted_id
        #return remap_labels(predicted_label)
            
    except Exception as e:
        print(f"Error inside predict_emotion_from_numpy: {e}")
        import traceback
        traceback.print_exc()
        return f"ERROR: {e}"
     
# --- These functions are helpers to get the pre-loaded objects ---
def getModel():
	return model
	
def getFeatureExtractor():
	return feature_extractor
	
def getId2label():
	return id2label