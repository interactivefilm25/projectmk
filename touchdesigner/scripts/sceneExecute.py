def onOffToOn(channel, sampleIndex, val, prev):
	return

def whileOn(channel, sampleIndex, val, prev):
	return

def onOnToOff(channel, sampleIndex, val, prev):
	return

def whileOff(channel, sampleIndex, val, prev):
	return

def onValueChange(channel, sampleIndex, val, prev):
	index = parent().par.Sceneindex
	print("Scene index:", index)

	if index == 0 and val == 1:
		print("play 00 ambient loop")
		op('00_Module').par.Play = 1
	elif index == 0 and val == 0:
		print("stop 00 ambient loop")
		# op('00_Module').par.Reload.pulse()
		op('00_Module').par.Play = 0
		run('restartAmbient()', delayFrames=50)
	return
	
def restartAmbient():
	print("restart 00 ambient loop")
	op('00_Module').par.Play = 1