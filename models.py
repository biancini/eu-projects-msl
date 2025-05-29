"""Data models for EU project document processing and management."""

from typing import Optional
from pydantic import BaseModel


class ProjetFileData(BaseModel):
    """Data model containing file paths for project-related file documents."""
    project_name: str
    base_path: str
    start_date: str
    call_file: str
    proposal_file: str
    ga_file: str


class ProjectData(BaseModel):
    """Data model representing a project's information and associated documents."""
    project_name: str
    start_date: str
    call_text: Optional[list[str]] = None
    proposal_text: Optional[list[str]] = None
    ga_text: Optional[list[str]] = None
