import opensmile
import numpy as np

audio_buffer_chop = op('nullAudio')
sampling_rate = op('audiodevin1').par.rate
output_table = op('table_features')

def onInitialize(timerOp, callCount):
	print('Init 2 OpenSMILE Analysis')
	
	smile = opensmile.Smile(
		feature_set=opensmile.FeatureSet.eGeMAPSv02,
	    #feature_set=opensmile.FeatureSet.ComParE_2016,
    	feature_level=opensmile.FeatureLevel.Functionals,
	)
	me.storage['smile_instance'] = smile
	
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

		output_table.clear()
		output_table.appendRow(features.columns)
		output_table.appendRow(features.iloc[0].tolist())
		#print(features.iloc[0].tolist())

	except Exception as e:
		print(f"Error during OpenSMILE processing: {e}")
	return

def onSubrangeStart(timerOp):
	return

	