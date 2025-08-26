import threading
import queue
import time
import datetime
import numpy as np
import json

_manager_instance = None

class EmotionPredictor:    
    def __init__(self, ownerComp):
        self.ownerComp = ownerComp
        self.tasks = queue.Queue(maxsize=10)
        self.results = queue.Queue()
        self.worker_thread = None
        self.initialized = False

        try:
            print("StreamingEmotionPredictor: Loading model module...")
            ml_module = mod('modelLoader')

            self.predict_func = ml_module.predict_emotion_from_numpy
            self.initialized = True
            self.start_worker()
            print("EmotionPredictor: Models and functions loaded successfully.")

        except Exception as e:
            print(f"CRITICAL INIT ERROR: Could not load from 'modelLoader'. Manager is disabled. Error: {e}")
            self.initialized = False

    def start_worker(self):
        if not self.initialized:
            print("Cannot start worker, manager is not initialized.")
            return
        if self.worker_thread and self.worker_thread.is_alive():
            return

        self.worker_thread = threading.Thread(
            target=self.worker_function,
            args=(self.predict_func,), 
            daemon=True
        )
        self.worker_thread.start()
        print("Worker thread started.")

    def worker_function(self, predict_func):
        chunk_id = 0

        while True:
            try:
                print(f"Worker (Thread ID: {threading.get_ident()}): Waiting for a task...")
                task = self.tasks.get()
                print(f"Worker (Thread ID: {threading.get_ident()}): Got a task!")
                
                if task is None: # Shutdown
                    print("Worker thread received shutdown signal. Exiting.")
                    break

                audio_array = task['array']
                sample_rate = task['rate']
            
                print(f"Worker: Processing chunk {chunk_id} (size: {len(audio_array)} samples)")
            
                start_time = time.time()
            
                result = predict_func(audio_array, sample_rate)
                print(result)

                end_time = time.time()
                duration = end_time - start_time
                print(f"Worker: Finished chunk {chunk_id} in {duration:.2f}s. Result: {result}")
                self.results.put({
                    "chunk_id": chunk_id, 
                    "result": result, 
                    "duration": duration
                })
                self.tasks.task_done()
                chunk_id += 1
            except Exception as e:
                print(f"Worker Thread Crashed!")
                import traceback
                traceback.print_exc()

    def add_numpy_task(self, audio_array, sample_rate):
        if not self.initialized:
            print("Cannot add task, manager is not initialized.")
            return False

        try:
            task = {'array': audio_array, 'rate': sample_rate}
            self.tasks.put_nowait(task)
            return True
        except queue.Full:
            print("Warning: Prediction queue is full. Skipping audio chunk.")
            return False

    def check_for_results(self):
        if not self.initialized:
            return
        try:
            result = self.results.get_nowait()
            print(f"Main Thread: Got result -> {result}")

            results_table = op('results_table')
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            emotion_json = json.dumps(result['result'], ensure_ascii=False)
            print(f"Main Thread: Appending result to table -> {emotion_json}")
            results_table.appendRow([result['chunk_id'], emotion_json, timestamp, f"{result['duration']:.2f}"])

            self.results.task_done()
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error retrieving result or updating table: {e}")
            
    def shutdown(self):
        if self.worker_thread and self.worker_thread.is_alive():
            print("Main Thread: Sending shutdown signal to worker.")
            self.tasks.put(None)
            self.worker_thread.join(timeout=2)
        self.initialized = False

def GetManager():
    global _manager_instance
    if _manager_instance is None:
        print("Manager instance not found, creating a new one.")
        _manager_instance = EmotionPredictor(me.parent())
    return _manager_instance

def ShutdownManager():
    global _manager_instance
    if _manager_instance is not None:
        print("Executing manager shutdown.")
        _manager_instance.shutdown()
        _manager_instance = None