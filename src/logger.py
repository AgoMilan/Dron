import csv, time

class CSVLogger:
    def __init__(self, path):
        self.path = path
        self.f = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.f)
        self.writer.writerow(["ts", "frame_idx", "track_id", "cx", "cy", "w", "h", "state", "pan_cmd", "tilt_cmd"])

    def log(self, row):
        row_out = [time.time()] + row
        self.writer.writerow(row_out)
        self.f.flush()

    def close(self):
        self.f.close()
