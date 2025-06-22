"""Load documents for each project and initialize the ChromaDB."""
import logging

from src.euprojectsrag.file_reader import FileReader
from src.euprojectsrag.configurations import get_project_conf
from src.euprojectsrag.data_models import PROJECT_LIST

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

reader = FileReader()

for project_name in PROJECT_LIST:
    project_conf = get_project_conf(project_name)
    reader.get_chroma_db(project_conf)

logging.info("Documents loaded for projects: %s", PROJECT_LIST)
