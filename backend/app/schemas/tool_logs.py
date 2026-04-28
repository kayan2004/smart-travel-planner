from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ToolLogRead(BaseModel):
    id: int
    agent_run_id: int
    tool_name: str
    input_payload: str
    output_payload: str
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
