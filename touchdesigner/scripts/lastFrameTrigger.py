def onOffToOn(channel, sampleIndex, val, prev):
	return

def whileOn(channel, sampleIndex, val, prev):
	return

def onOnToOff(channel, sampleIndex, val, prev):
	return

def whileOff(channel, sampleIndex, val, prev):
	return

def onValueChange(channel, sampleIndex, val, prev):
	clip = op('clip')
	
	if (channel.name == 'last_frame' and val == 1.0):
		clip.par.cuepulse.pulse()
		clip.par.play = False
		
		parent().par.Playstate = False
		# parent().par.Play = 0
		# print('Stop media in', parent().name)

	return
	