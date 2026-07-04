from pydantic import BaseModel, Field


class ClusterNamingProposal(BaseModel):
    """Claude's proposed human-readable label for one HDBSCAN cluster."""

    tag_name: str = Field(min_length=1, max_length=100)
    description: str = Field(min_length=1, max_length=1000)
