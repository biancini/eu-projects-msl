"""Data models for EU project document processing and management."""

from typing import Literal, List

from pydantic import BaseModel, Field

PROJECT_LIST = [
    "SPECTRO", "EMAI4EU", "RESCHIP4EU", "ACHIEVE",
]

class ProjetFileData(BaseModel):
    """Data model containing file paths for project-related file documents."""
    project_name: str = Field(description="Name of the project")
    base_path: str = Field(description="Base path for the project")
    start_date: str = Field(description="Start date of the project")
    call_file: str = Field(description="Path to the call file")
    proposal_file: str = Field(description="Path to the proposal file")
    ga_file: str = Field(description="Path to the grant agreement file")


class ProjectData(BaseModel):
    """Data model representing a project's information and associated documents."""
    project_name: str = Field(description="Name of the project")
    start_date: str = Field(description="Start date of the project")
