from pydantic import BaseModel, Field, ValidationError
from json import JSONDecodeError, load
from typing import Any

class ParameterType(BaseModel):
    type: str = Field(min_length=1)

class FuncDef(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParameterType]|None = Field(default=None)
    returns: ParameterType|None = Field(default=None)

class PromptType(BaseModel):
    prompt: str = Field(min_length=1)

def parse_func_defs(filepath: str) -> list[FuncDef]:
    try:
        with open(filepath, "r") as file_obj:
            data: Any = load(file_obj)
    except ValidationError as e:
        print(f"Validation error: {e}")
    except JSONDecodeError as e:
        print(f"JSON format error: {e}")

    param_type = ParameterType()
    f_def = FuncDef()
    for key, val in data:

