def onStart():
    print('----------------------------------------------------')
    print(f'SUCCESS: AscendingIntelligence project started at {absTime.frame}')
    print('----------------------------------------------------')

    # Ensure any old manager from a previous session/script-recompile is gone.
    mod('queueManager').ShutdownManager()

    # Run the model loader to ensure models are ready.
    op('modelLoader').run()
    return

def onCreate():
    return

def onExit():
    # Call dedicated shutdown function when the project closes.
    print("Project is closing, shutting down the emotion predictor manager.")
    mod('queueManager').ShutdownManager()
    return

def onFrameStart(frame):
    return

def onFrameEnd(frame):
    mod('queueManager').GetManager().check_for_results()
    return

def onPlayStateChange(state):
    return

def onDeviceChange():
    return

def onProjectPreSave():
    return

def onProjectPostSave():
    return