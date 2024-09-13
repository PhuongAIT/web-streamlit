from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
#WEBCAM = 'Webcam'
#RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, YOUTUBE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'class.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'class_detected.jpg'


VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'video_1': VIDEO_DIR / 'video_1.mp4',
    'video_2': VIDEO_DIR / 'video_2.mp4',
    'video_3': VIDEO_DIR / 'video_3.mp4',
    'video_4': VIDEO_DIR / 'video_4.mp4',
    'video_5': VIDEO_DIR / 'video_5.mp4',
    # 'video_6': VIDEO_DIR / 'video_6.mp4',
    # 'video_7': VIDEO_DIR / 'video_7.mp4',
    # 'video_8': VIDEO_DIR / 'video_8.mp4',
    #'video_9': VIDEO_DIR / 'video_9.mp4',
}



MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'best.pt'



# Webcam
#WEBCAM_PATH = 0
