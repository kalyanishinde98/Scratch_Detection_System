
from pathlib import Path
import xml.etree.ElementTree as ET
import random
import shutil
import yaml

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "NEU-DET"
OUTPUT_DIR = PROJECT_ROOT / "data" / "scratch_yolo"

CLASS_NAME = "scratches"
CLASS_ID = 0

RANDOM_SEED = 42

# Keep original train and validation groups separate.
# Divide original training data into train/test.
TRAIN_RATIO = 0.85

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


# --------------------------------------------------
# FIND IMAGE FILE
# --------------------------------------------------


def find_image(image_name, image_directories):

    image_filename = Path(image_name).name
    image_stem = Path(image_name).stem

    for directory in image_directories:

        # First check the exact filename recursively.
        for candidate in directory.rglob(image_filename):
            if candidate.is_file():
                return candidate

        # If the XML filename has no extension,
        # search by image filename without extension.
        for candidate in directory.rglob("*"):

            if not candidate.is_file():
                continue

            if candidate.suffix.lower() not in [
                ".jpg", ".jpeg", ".png", ".bmp"
            ]:
                continue

            if candidate.stem.lower() == image_stem.lower():
                return candidate

    return None



# --------------------------------------------------
# CONVERT XML BOX TO YOLO FORMAT
# --------------------------------------------------

def convert_box_to_yolo(box, image_width, image_height):

    xmin = float(box.findtext("xmin"))
    ymin = float(box.findtext("ymin"))
    xmax = float(box.findtext("xmax"))
    ymax = float(box.findtext("ymax"))

    # Clip coordinates to image boundaries.
    xmin = max(0, min(xmin, image_width))
    xmax = max(0, min(xmax, image_width))
    ymin = max(0, min(ymin, image_height))
    ymax = max(0, min(ymax, image_height))

    if xmax <= xmin or ymax <= ymin:
        return None

    x_center = ((xmin + xmax) / 2) / image_width
    y_center = ((ymin + ymax) / 2) / image_height

    width = (xmax - xmin) / image_width
    height = (ymax - ymin) / image_height

    return (
        f"{CLASS_ID} "
        f"{x_center:.6f} "
        f"{y_center:.6f} "
        f"{width:.6f} "
        f"{height:.6f}"
    )


# --------------------------------------------------
# READ XML ANNOTATION
# --------------------------------------------------

def read_annotation(xml_path):

    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")

    size = root.find("size")

    if size is None:
        raise ValueError(f"Missing image size in {xml_path}")

    image_width = int(size.findtext("width"))
    image_height = int(size.findtext("height"))

    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image dimensions in {xml_path}")

    labels = []
    difficult_scratch_found = False
    scratch_found = False

    for obj in root.findall("object"):

        name = obj.findtext("name", "").strip().lower()

        if name != CLASS_NAME:
            continue

        scratch_found = True

        difficult = obj.findtext("difficult", "0").strip()

        if difficult == "1":
            difficult_scratch_found = True
            continue

        box = obj.find("bndbox")

        if box is None:
            continue

        label = convert_box_to_yolo(
            box,
            image_width,
            image_height
        )

        if label:
            labels.append(label)

    return filename, labels, scratch_found, difficult_scratch_found


# --------------------------------------------------
# PREPARE ONE DATA SPLIT
# --------------------------------------------------

def prepare_split(xml_directory, image_directory, split_name):

    xml_files = list(xml_directory.glob("*.xml"))

    processed = 0
    positive = 0
    difficult_only = 0
    invalid = 0

    records = []

    for xml_path in xml_files:

        try:

            filename, labels, scratch_found, difficult_found = (
                read_annotation(xml_path)
            )

            if not filename:
                invalid += 1
                continue

            image_path = find_image(
                filename,
                [image_directory]
            )

            if image_path is None:
                invalid += 1
                print(f"Missing image: {filename}")
                continue

            # Avoid using an image with only difficult
            # scratch objects as a negative training example.
            if not labels and difficult_found:
                difficult_only += 1
                continue

            records.append((image_path, labels))

        except Exception as error:

            invalid += 1
            print(f"Error reading {xml_path.name}: {error}")

    # Deterministic shuffle.
    random.Random(RANDOM_SEED).shuffle(records)

    # Split only original training data into train and test.
    if split_name == "train":

        split_index = int(len(records) * TRAIN_RATIO)

        train_records = records[:split_index]
        test_records = records[split_index:]

        write_records(train_records, "train")
        write_records(test_records, "test")

        processed = len(records)
        positive = sum(bool(labels) for _, labels in records)

    else:

        write_records(records, "val")

        processed = len(records)
        positive = sum(bool(labels) for _, labels in records)

    print(f"\n{split_name.upper()} DATA")
    print("-----------------------")
    print("Processed:", processed)
    print("Images with scratch labels:", positive)
    print("Negative images:", processed - positive)
    print("Excluded difficult-only images:", difficult_only)
    print("Invalid or missing files:", invalid)


# --------------------------------------------------
# WRITE IMAGES AND LABELS
# --------------------------------------------------

def write_records(records, split_name):

    image_output = OUTPUT_DIR / "images" / split_name
    label_output = OUTPUT_DIR / "labels" / split_name

    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    for image_path, labels in records:

        destination_image = image_output / image_path.name

        shutil.copy2(image_path, destination_image)

        label_path = label_output / (
            image_path.stem + ".txt"
        )

        label_path.write_text(
            "\n".join(labels),
            encoding="utf-8"
        )


# --------------------------------------------------
# CREATE YOLO DATASET YAML
# --------------------------------------------------

def create_data_yaml():

    data_config = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "scratch"
        }
    }

    yaml_path = OUTPUT_DIR / "data.yaml"

    with open(yaml_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(
            data_config,
            file,
            sort_keys=False
        )

    print("\nYOLO configuration created:")
    print(yaml_path)


# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main():

    print("SCRATCH DETECTION - DATASET PREPARATION")
    print("=======================================")

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DIR}"
        )

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    train_xml = RAW_DIR / "train" / "annotations"
    train_images = RAW_DIR / "train" / "images"

    val_xml = RAW_DIR / "validation" / "annotations"
    val_images = RAW_DIR / "validation" / "images"

    for directory in [
        train_xml,
        train_images,
        val_xml,
        val_images
    ]:

        if not directory.exists():
            raise FileNotFoundError(
                f"Required folder missing: {directory}"
            )

    prepare_split(
        train_xml,
        train_images,
        "train"
    )

    prepare_split(
        val_xml,
        val_images,
        "validation"
    )

    create_data_yaml()

    print("\n=======================================")
    print("DATASET PREPARATION COMPLETED")
    print("=======================================")


if __name__ == "__main__":
    main()
