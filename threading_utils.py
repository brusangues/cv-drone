import cv2
import queue
import threading
import time

# bufferless VideoCapture
class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name, cv2.CAP_DSHOW)
        self.q = queue.Queue()
        self.t = threading.Thread(target=self._reader)
        #t.daemon = True
        self.t.start()
        
    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("break")
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)
            self.q.task_done()

    def read(self):
        frame = self.q.get()
        return frame
    
    def release(self):
        self.cap.release()
        self.q.join()
        self.q.put(None)
        self.t.join()
        print("released")