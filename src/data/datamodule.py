import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def check_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()

        with Image.open(file_path) as img:
            img.transpose(Image.FLIP_LEFT_RIGHT)
            return None

    except (IOError, SyntaxError, OSError) as e:
        print(f"Corrupted file found: {file_path} ({e})")
        return file_path


def clean_dataset(root_dir):
    root = Path(root_dir)
    files = list(root.rglob("*.jpg")) + list(root.rglob("*.png")) + list(root.rglob("*.jpeg"))

    print(f"Scanning {len(files)} images for corruption...")

    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(check_image, files), total=len(files)))

    bad_files = [f for f in results if f is not None]

    print(f"Found {len(bad_files)} corrupted images.")
    for f in bad_files:
        try:
            os.remove(f)
            print(f"Deleted: {f}")
        except OSError as e:
            print(f"Failed to delete: {f} ({e})")


if __name__ == "__main__":
    clean_dataset("data/raw")