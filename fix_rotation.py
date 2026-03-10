from PIL import Image, ExifTags
import os

DATASET = "dataset_detection/images"

def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break

        exif = image._getexif()

        if exif is not None:
            orientation_value = exif.get(orientation)

            if orientation_value == 3:
                image = image.rotate(180, expand=True)
            elif orientation_value == 6:
                image = image.rotate(270, expand=True)
            elif orientation_value == 8:
                image = image.rotate(90, expand=True)

    except:
        pass

    return image


for file in os.listdir(DATASET):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(DATASET, file)

        img = Image.open(path)
        img = correct_orientation(img)
        img.save(path)

print("✅ All images orientation fixed!")