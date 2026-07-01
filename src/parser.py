from pydantic import BaseModel, Field, ValidationError
from json import JSONDecodeError, load
from typing import Any
import sys


class ParameterType(BaseModel):
    """Pydantic model representing a typed function parameter."""
    type: str = Field(min_length=1)
    properties: dict[str, 'ParameterType'] | None = Field(default=None)


ParameterType.model_rebuild()


class FuncDef(BaseModel):
    """Pydantic model representing a function definition."""
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: dict[str, ParameterType] | None = Field(default=None)
    returns: ParameterType | None = Field(default=None)


class PromptDef(BaseModel):
    """Pydantic model representing a user prompt entry."""
    prompt: str = Field(min_length=1)


def parse_func_defs(filepath: str) -> list[FuncDef]:
    """Parse a JSON file of function definitions.

    Args:
        filepath: Path to the JSON file containing function definitions.

    Returns:
        A list of validated FuncDef objects.

    Raises:
        SystemExit: If the file cannot be read, is invalid JSON,
            or fails pydantic validation.
    """
    try:
        with open(filepath, "r") as file_obj:
            data: list[dict[str, Any]] = load(file_obj)
    except (FileNotFoundError, PermissionError) as e:
        print(f"File error: {e}", file=sys.stderr)
        sys.exit(1)
    except JSONDecodeError as e:
        print(f"JSON format error: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    for item in data:
        try:
            f_def = FuncDef(**item)
            results.append(f_def)
        except ValidationError as e:
            print(f"Validation error: {e}")
            sys.exit(1)
    return results


def parse_prompts(filepath: str) -> list[PromptDef]:
    """Parse a JSON file of user prompts.

    Args:
        filepath: Path to the JSON file containing prompt entries.

    Returns:
        A list of validated PromptDef objects.

    Raises:
        SystemExit: If the file cannot be read, is invalid JSON,
            or fails pydantic validation.
    """
    try:
        with open(filepath, "r") as prompt_file_obj:
            data: list[dict[str, str]] = load(prompt_file_obj)
    except (FileNotFoundError, PermissionError) as e:
        print(f"File error: {e}", file=sys.stderr)
        sys.exit(1)
    except JSONDecodeError as e:
        print(f"JSON format error: {e}", file=sys.stderr)
        sys.exit(1)

    results = []
    for item in data:
        try:
            p_def = PromptDef(**item)
            results.append(p_def)
        except ValidationError as e:
            print(f"Validation error: {e}", file=sys.stderr)
            sys.exit(1)
    return results
