
from models import ProjetFileData
from euprojects import read_pdf_pages
import matplotlib.pyplot as plt


BASEPATH = (
    "/Users/andrea.biancinigmail.com/Library/CloudStorage/OneDrive-Personale/"
    "Documenti/work/eit digital/"
)

spectro_conf = ProjetFileData(
    project_name="SPECTRO",
    start_date="2023-09-01",
    base_path=f"{BASEPATH}/Projects/SPECTRO/",
    call_file="call-fiche_digital-2022-skills-03-specialised-edu_en.pdf",
    proposal_file="Proposal-SEP-210919492_SPECTRO.pdf",
    ga_file="Agreements/101123118_Amendment/Amendment - AMD-101123118-2.pdf"
)

call_text = read_pdf_pages(spectro_conf.base_path + spectro_conf.call_file)
proposal_text = read_pdf_pages(spectro_conf.base_path + spectro_conf.proposal_file)
ga_text = read_pdf_pages(spectro_conf.base_path + spectro_conf.ga_file)

all_pages = call_text + proposal_text + ga_text
char_counts = [len(page) for page in all_pages]

plt.figure(figsize=(10, 6))
plt.hist(char_counts, bins=20, edgecolor='black')
plt.title('Distribution of Words per Page')
plt.xlabel('Number of Documents')
plt.ylabel('Number of Words')
plt.grid(True, alpha=0.3)
plt.show()