def onInitialize(timerOp, callCount):
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
	audio_chop = op('audioBuffer')
	
	numpy_array = audio_chop.numpyArray()[0].copy()
	sample_rate = audio_chop.rate
	
	manager = mod('queueManager').GetManager()
	was_added = manager.add_numpy_task(numpy_array, sample_rate)
	
	if (was_added):
		print("Main Thread: Sent 2s audio chunk to worker.")
	return

def onDone(timerOp, segment, interrupt):
	return

def onSubrangeStart(timerOp):
	return

	