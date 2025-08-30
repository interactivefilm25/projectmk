def onOffToOn(channel, sampleIndex, val, prev):
	return

def whileOn(channel, sampleIndex, val, prev):
	return

def onOnToOff(channel, sampleIndex, val, prev):
	return

def whileOff(channel, sampleIndex, val, prev):
	return

def onValueChange(channel, sampleIndex, val, prev):
	if val == 1.0:
		print("Presence Detected")
		parent().par.Sceneindex = 1
	else:
		print("No Presence")
		parent().par.Sceneindex = 0
	return
	