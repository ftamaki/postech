from dataclasses import dataclass, field
from typing import List
from uuid import UUID


@dataclass
class ProcessingJob:
    analysis_id: UUID
    file_path: str
    file_type: str
    original_filename: str


@dataclass
class Component:
    name: str
    type: str
    description: str


@dataclass
class Risk:
    severity: str  # high | medium | low
    title: str
    description: str


@dataclass
class Recommendation:
    priority: str  # high | medium | low
    title: str
    description: str


@dataclass
class AIAnalysisResult:
    analysis_id: UUID
    components: List[Component] = field(default_factory=list)
    risks: List[Risk] = field(default_factory=list)
    recommendations: List[Recommendation] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""
