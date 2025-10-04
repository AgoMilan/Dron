
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, **kw):
        # pass through DeepSort parameters
        self.tracker = DeepSort(**kw)

    def update(self, dets, frame=None):
        # dets format: [([x,y,w,h], conf, cls), ...]
        tracks = self.tracker.update_tracks(dets, frame=frame)
        return tracks
