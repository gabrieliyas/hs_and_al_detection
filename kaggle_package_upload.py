from pathlib import Path
import shutil

# Change this to your actual local repository path.
REPO = Path(r"D:\repos\hs_and_al_detection")

# Temporary directory for preparing Kaggle uploads.
UPLOAD = Path(r"D:\repos")

UPLOAD.mkdir(parents=True, exist_ok=True)

# Define source and staging directories.
data_source = REPO / "data"

data_stage = UPLOAD / "hs-and-al-detection-data"

# Check that the source directories exist.
assert data_source.is_dir(), f"Missing directory: {data_source}"

# Remove previous staging copies to avoid uploading stale files.
for directory in (data_stage):
    if directory.exists():
        shutil.rmtree(directory)

# Preserve the original directory structures.
shutil.copytree(data_source, data_stage / "data")

# Create ZIP archives.
data_zip = shutil.make_archive(
    str(UPLOAD / "hs-and-al-detection-data"),
    "zip",
    root_dir=data_stage
)

print("Created upload packages:")
print(data_zip)