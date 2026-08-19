
from pydantic import BaseModel



class APIResponse(BaseModel):
    success: bool
    message: str

class InputData(BaseModel):
    input_str: str
