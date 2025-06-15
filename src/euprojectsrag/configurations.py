"""Configuration settings and project data for EU projects."""

from .data_models import ProjetFileData

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

emai4eu_conf = ProjetFileData(
    project_name="EMAI4EU",
    start_date="2024-01-01",
    base_path=f"{BASEPATH}/Projects/EMAI4EU/",
    call_file="call-fiche_digital-2022-skills-03-specialised-edu_en.pdf",
    proposal_file="Proposal-SEP-210919498.pdf",
    ga_file="Agreements/101123289_Amendment/Amendment - AMD-101123289-2.pdf"
)

reschip4eu_conf = ProjetFileData(
    project_name="RESCHIP4EU",
    start_date="2024-06-01",
    base_path=f"{BASEPATH}/Projects/RESCHIP4EU/",
    call_file="call-fiche_digital-2023-skills-04_en.pdf",
    proposal_file="PROPOSAL_101158828-RESCHIP4EU-DIGITAL-2023-SKILLS-04.pdf",
    ga_file="Agreements/Grant Agreement - GAP-101158828.pdf"
)

achieve_conf = ProjetFileData(
    project_name="ACHIEVE",
    start_date="2024-10-01",
    base_path=f"{BASEPATH}/Projects/ACHIEVE/",
    call_file="call-fiche_digital-2023-skills-05_en.pdf",
    proposal_file="Proposal-SEP-211044709.pdf",
    ga_file="Agreements/Grant Agreement - GAP-101190015.pdf"
)

arise_conf = ProjetFileData(
    project_name="ARISE",
    start_date="2025-07-01",
    base_path=f"{BASEPATH}/Projects/ARISE/",
    call_file="545_call-fiche_digital-2024-advanced-digital-07-keycapacity_en.pdf",
    proposal_file="ARISE_Proposal.pdf",
    ga_file="Agreements/Grant Agreement - GAP-101225600.pdf"
)

def get_project_conf(project_name: str) -> ProjetFileData:
    """Get the configuration for a project."""
    if project_name == "SPECTRO":
        return spectro_conf
    if project_name == "EMAI4EU":
        return emai4eu_conf
    if project_name == "RESCHIP4EU":
        return reschip4eu_conf
    if project_name == "ACHIEVE":
        return achieve_conf
    if project_name == "ARISE":
        return arise_conf

    raise ValueError(f"Project {project_name} not found")
