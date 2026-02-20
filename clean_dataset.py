from PIL import Image
import os

DATASET_PATH = "dataset"
valid_ext = (".jpg", ".jpeg", ".png")

removed = 0

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if not file.lower().endswith(valid_ext):
            continue

        path = os.path.join(root, file)
        try:
            img = Image.open(path)
            img.verify()   # verify image integrity
        except:
            os.remove(path)
            removed += 1
            print("Removed corrupted:", path)

print("Cleanup done.")
print("Total removed images:", removed)


#remove corrupted / fake images

#keep only valid eye / conjunctiva images

#permanently fix this error