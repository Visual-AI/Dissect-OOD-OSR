# import PIL
# from PIL import UnidentifiedImageError
# import glob

# imgs_ = glob.glob("/disk/datasets/clip_data/*/*.jpg")

# for img in imgs_:
#     try:
#         img = PIL.Image.open(img)
#     except PIL.UnidentifiedImageError:
#         print(img)


import PIL
from pathlib import Path
from PIL import UnidentifiedImageError
from PIL import Image

path = Path("/disk/datasets/clip_data").rglob("*.jpg")
for img_p in path:
    try:
        img = Image.open(img_p)
    except PIL.UnidentifiedImageError:
        print(img_p)