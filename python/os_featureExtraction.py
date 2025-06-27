import opensmile
import numpy as np
import joblib
import os.path

audio_buffer_chop = op('nullAudio')
sampling_rate = op('audiodevin1').par.rate
output_table = op('table_features')
env_params = op('APIKeys')

def onInitialize(timerOp, callCount):
	clear()
	print('Init 2 OpenSMILE Analysis')

	model_path = os.path.relpath(env_params['OPENSMILE_MODEL_PATH', 1].val)
	# model_path = r"C:\Users\Dev\Documents\Projects\KarenPalmer\AscendingIntelligence\python\model.pkl"
	# with open(model_path, 'rb') as f:
	# 	me.storage['model'] = f.read()
	# 	print(f"Model file {model_path} opened successfully.")
	# return 0

	# conf_path = os.path.relpath(env_params['OPENSMILE_EMOBASE_CONF', 1].val)
	# model_path = os.path.relpath(env_params['OPENSMILE_MODEL_PATH', 1].val)

	smile = opensmile.Smile(
		feature_set=opensmile.FeatureSet.emobase,
    	feature_level=opensmile.FeatureLevel.Functionals,
	)
		#feature_set=opensmile.FeatureSet.eGeMAPSv02,
	    #feature_set=opensmile.FeatureSet.ComParE_2016,
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

	try:
		features = smile.process_signal(
			signal=signal_mono,
			sampling_rate=sampling_rate
		)

		trained_cols = model.feature_names_in_
		X = features.to_numpy().reshape(1, -1)
		X_df = np.zeros((1, len(trained_cols)))
		for i, feat in enumerate(features.columns):
			if feat in trained_cols:
				idx = list(trained_cols).index(feat)
				X_df[0, idx] = X[0, i]
		probs = model.predict_proba(X_df)[0]
		emo = model.classes_[np.argmax(probs)]

		# print(X)
		# print(f"Feature names from model: {trained_cols}")
		# print(emo)
		
		output_table.clear()
		output_table.appendRow(emo)
		# output_table.appendRow(features.iloc[0].tolist())
		#print(features.iloc[0].tolist())

	except Exception as e:
		print(f"Error during OpenSMILE processing: {e}")
	return

def onSubrangeStart(timerOp):
	return

	