@echo off
call venv\Scripts\activate
python .\scripts\track_drone22.py --source test_videos\drone1.mp4 --show --winsize 640 --pause_frame 20 --max_age 60 --n_init 3 --iou_dist 0.7
