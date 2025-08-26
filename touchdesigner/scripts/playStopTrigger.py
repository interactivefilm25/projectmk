from numpy import clip


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
	params = op('parameters')
	
	if channel.name == 'Loopcontent':
		clip.par.textendright = val

	if channel.name == 'Play' and val == 1.0:
		clip.par.cuepulse.pulse()
		clip.par.play = True
		
		parent().par.Playstate = True
		
		print('Play media in', parent().name)

	if channel.name == 'Stop' and val == 1.0:
		clip.par.cuepulse.pulse()
		clip.par.play = False

		parent().par.Playstate = False
		
		print('Stop media in', parent().name)
		
		
	return
	