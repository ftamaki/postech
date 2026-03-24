from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from uuid import UUID


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
class Report:
    id: UUID
    analysis_id: UUID
    components: List[Component]
    risks: List[Risk]
    recommendations: List[Recommendation]
    summary: str
    raw_ai_response: str
    created_at: datetime
