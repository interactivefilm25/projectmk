import opensmile
import joblib
import numpy as np
import pandas as pd

audio_buffer_chop = op('nullAudio')
sampling_rate = op('audiodevin1').par.rate
output_table = op('table_features')
env_params = op('APIKeys')

def onInitialize(timerOp, callCount):
	clear()
	print('Init 2 OpenSMILE Analysis')

	model_path = os.path.relpath(env_params['OPENSMILE_MODEL_PATH', 1].val)

	smile = opensmile.Smile(
		feature_set=opensmile.FeatureSet.emobase,
    	feature_level=opensmile.FeatureLevel.Functionals,
	)
	me.storage['smile_instance'] = smile
	me.storage['model'] = joblib.load(model_path)

	print(f"Successfully initialized with FeatureSet: {smile.feature_set.name}")
	print(f"Successfully initialized with FeatureLevel: {smile.feature_level.name}")
	return 0

def onReady(timerOp):
	return
	
def onStart(timerOp):
	return
	
def onTimerPulse(timerOp, segment):
	return

def whileTimerActive(timerOp, segment, cycle, fraction):
	return

def onSegmentEnter(timerOp, segment, interrupt):
	return
	
def onSegmentExit(timerOp, segment, interrupt):
	return

def onCycleStart(timerOp, segment, cycle):
	return

def onCycleEndAlert(timerOp, segment, cycle, alertSegment, alertDone, interrupt):
	return
	
def onCycle(timerOp, segment, cycle):
	return

def onDone(timerOp, segment, interrupt):
	smile = me.storage.get('smile_instance')
	model = me.storage.get('model')
	
	if smile is None:
		print("Error: Smile instance not found.")
		return

	source_chop = audio_buffer_chop
	np_array = source_chop.numpyArray()
	
	signal_mono = np_array[0]

	emo, probs = predict(signal_mono)
	print(f"Predicted Emotion: {emo}")
	for index, (emotion, prob) in enumerate(probs.items()):
		output_table[index, 0] = emotion
		output_table[index, 1] = prob/100.
		print(f"{index} {emotion:9s}: {prob:6.2f}%")

	return 0

def onSubrangeStart(timerOp):
	return

def predict(signal):
	smile = me.storage.get('smile_instance')
	model = me.storage.get('model')

	features = smile.process_signal(
		signal=signal,
        sampling_rate=sampling_rate
    )

	trained_cols = model.feature_names_in_
	aligned_features = features.reindex(columns=trained_cols, fill_value=0.0)    
	
	probs = model.predict_proba(aligned_features)[0]
	emo = model.classes_[np.argmax(probs)]

	return emo, dict(zip(model.classes_, (probs * 100).round(2)))
