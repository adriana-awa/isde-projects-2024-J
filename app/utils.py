import os
from pathlib import Path
from app.config import Configuration

conf = Configuration()


def list_images():
    """Returns the list of available images."""
    img_names = filter(
        lambda x: x.endswith(".JPEG"), os.listdir(conf.image_folder_path)
    )
    return list(img_names)

def get_temp_file_path(filename: str) -> str:
    """Returns the correct path for temporary uploaded files."""
    base_path = Path(__file__).parent
    temp_dir = base_path / 'static' / 'temp'
    
    # Create temp directory if it doesn't exist
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    return str(temp_dir / filename)