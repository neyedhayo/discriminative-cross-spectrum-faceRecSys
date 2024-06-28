import zipfile
import cv2
import numpy as np
import os
from io import BytesIO
from tqdm import tqdm
from PIL import Image

# zip path
file_path = '../../data/raw/face Dataset.zip'
processed_data = ''