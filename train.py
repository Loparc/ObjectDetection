from converter import read_content
DIR_TO_XML_FILE = "/dtu/datasets1/02516/potholes/annotations"
from pathlib import Path

folder = Path(DIR_TO_XML_FILE)

for file_path in folder.iterdir():

    if file_path.is_file():
        content = file_path.read_text(encoding="utf-8", errors="replace")
        name, boxes = read_content(content)
    