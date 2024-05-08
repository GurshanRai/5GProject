Dependencies:
- Python 3.10.x
- Pip modules listed under requirements.txt
- ffmpeg installed (See ffmpeg install guide below)

How to install ffmpeg on Windows without adminstrator access:
    1. Download ffmpeg-git-essentials zip file (https://www.gyan.dev/ffmpeg/builds/)
    2. Extract files and place the bin folder inside of your working directory (same directory containing .py files)

Instructions on how to run:
- Ensure field.jpg used for line homography is in working directory (same directory as line_detection.py)
- Ensure model.tflite used for detection model is also in working directory
- Program needs 3 command line arguments when being executed:
    Usage: python3 line_detection <videos folder> <output folder for line detection> <output folder for ball tracking>
      - Videos folder: Folder containing 4 videos
      - Output folder for line detection: Empty folder to hold videos outputted by line detection processing
      - Output folder for ball tracking: Empty folder to hold videos outputted by ball tracking processing
- Line Detection:
    1. Click 4 points in pop up window of first video frame and press ESC to continue
    2. Click 4 points in second pop up window of field.jpg and press ESC to continue
    3. Wait for program to process line detection for entire video (wait can vary depending on video length)
    4. Repeat for other 3 videos
- Ball Tracking:
    1. Allow program to process entire video (wait can vary depending on video length)
    2. If program detected or is tracking ball incorrectly, press the 1, 2, 3, or 4 number key to switch a video from tracking to detecting mode
        (1 switches top left window, 2 switches top right window, 3 switches bottom left window, and 4 switches bottom right window)

Warnings/possible errors:
- Videos must be in .mp4 format (required by line detection code)
- Videos must be around the same length (as ball tracking will stop when shortest video ends)
- There must be exactly 4 videos in the videos folder (required by ball tracking code)
- Video names must contain 0 whitespace (required by ball tracking code)
- The 3 folders provided in the command line arguments must exist
- _temp folder does not exist in working directory (required by ball tracking code)
