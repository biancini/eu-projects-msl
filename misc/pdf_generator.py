"""Generates a list of documents from a PDF file and print them."""
from euprojectsrag.readpdfs import get_documents_from_pdf
from euprojectsrag.confs import get_project_conf

project_conf = get_project_conf("SPECTRO")

ga_docs = get_documents_from_pdf(
    project_conf.base_path + project_conf.ga_file,
    "Grant Agreement",
    project_conf.project_name
)

for doc in ga_docs:
    print(doc.page_content)
