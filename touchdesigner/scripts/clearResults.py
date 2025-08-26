def onOffToOn(channel, sampleIndex, val, prev):
    results = op('results_table')
    results.clear()
    results.appendRow(['chunk_id', 'emotion', 'timestamp', 'duration'])
    results.appendRow([-1, {}, -1, -1])
    return

def whileOn(channel, sampleIndex, val, prev):
    return

def onOnToOff(channel, sampleIndex, val, prev):
    return

def whileOff(channel, sampleIndex, val, prev):
    return

def onValueChange(channel, sampleIndex, val, prev):
    return
