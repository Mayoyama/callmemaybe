from json import dump
from typing import Any
import sys


def print_step(label: str, current: str, logits: list[float],
               selected: str, decode_dict: dict[int, str]) -> None:
    """Print the top 3 token candidates and selected token for
    a decoding step.

    Args:
        label: Prefix label (function name or parameter name).
        current: The string built so far in this generation pass.
        logits: Full logit list from the model.
        selected: The token that was selected.
        decode_dict: A dict mapping token IDs to their string
        representations.
    """
    top = sorted(
        ((i, s) for i, s in enumerate(logits) if i in decode_dict),
        key=lambda x: x[1],
        reverse=True
    )[:3]
    top_tokens = [(decode_dict[i], round(s, 4)) for i, s in top]
    print(f"  {label} [{current!r}] top: {top_tokens} → {selected!r}")


def gen_output_file(path: str, results: list[dict[str, Any]]) -> None:
    """Write results to a JSON output file.

    Args:
        path: Destination file path.
        results: List of result dictionaries to serialise.
    """
    try:
        with open(path, "w") as file:
            dump(results, file, indent=4)
    except OSError as e:
        print(f"Error - unable to save file: {e}", file=sys.stderr)


def recursive_gen_object_param(prompt: str, func: Any, props: dict[str, Any],
                               params: dict[str, Any],
                               small_llm: Any) -> dict[str, Any]:
    """Recursively extract values for a nested object parameter.

    Iterates over each sub-field in props. If a sub-field has type
    'object', recurses into its properties. Otherwise delegates to
    gen_param_values and applies numeric type coercion.

    Args:
        prompt: The original natural language prompt.
        func: The function definition being processed.
        props: A dict mapping sub-field names to their ParameterType.
        params: Accumulated parameter values extracted so far.
        small_llm: The LLM wrapper used for constrained decoding.

    Returns:
        A dict mapping each sub-field name to its extracted value.
    """
    prop_dict: dict[str, Any] = {}
    for sub_name, sub_param in props.items():
        if sub_param.type == 'object':
            if sub_param.properties is not None:
                prop_dict[sub_name] = (recursive_gen_object_param(prompt, func,
                                       sub_param.properties, params,
                                       small_llm))
            else:
                print(f"[recursive_gen_object_param()]: '{sub_name}' has type "
                      f"'object' but no properties defined", file=sys.stderr)
            continue
        raw = small_llm.gen_param_values(
            prompt, func, sub_name, sub_param.type, params
        )
        if sub_param.type not in ["number", "float", "int", "integer"]:
            prop_dict[sub_name] = raw
        else:
            try:
                if sub_param.type in ["number", "float"]:
                    value = float(raw)
                    prop_dict[sub_name] = (
                        int(value) if value.is_integer() else value
                    )
                elif sub_param.type in ["int", "integer"]:
                    prop_dict[sub_name] = int(float(raw))
            except ValueError:
                prop_dict[sub_name] = raw
    return prop_dict
