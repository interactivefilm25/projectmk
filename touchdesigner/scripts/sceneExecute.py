index = 0

def onOffToOn(channel, sampleIndex, val, prev):
	return

def whileOn(channel, sampleIndex, val, prev):
	return

def onOnToOff(channel, sampleIndex, val, prev):
	return

def whileOff(channel, sampleIndex, val, prev):
	return

def onValueChange(channel, sampleIndex, val, prev):
	if index == 0 and val == 1:
		print("play 00 ambient loop")
	elif index == 0 and val == 0:
		print("restart 00 ambient loop")
		# op('00_Module').par.Reload.pulse()
		# op('00_Module').par.Play = 0
		op('00_Module').par.Play = 1
	return
	