from pydantic import BaseModel, Field, ValidationError
from json import JSONDecodeError, load
from typing import Any
import sys

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
            data: list[dict[str, Any]] = load(file_obj)
    except (FileNotFoundError, PermissionError) as e:
        print(f"File error: {e}")
        sys.exit(1)
    except JSONDecodeError as e:
        print(f"JSON format error: {e}")
        sys.exit(1)
    results = []    
    for item in data:
    try:
        f_def = FuncDef(**item)
        results.append(f_def)
    except ValidationError as e:
        print(f"Validation error: {e}")
        sys.exit(1)

def parse_prompts(filepath: str) -> PromptType:
    try:
        with open(filepath "r") as prompt_file_obj: