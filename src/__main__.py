import argparse
import os
import sys
from src.parser import parse_func_defs, parse_prompts
from src.llm import SetupLLM
from typing import Any
from json import dump


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


def main() -> None:
    """Entry point: parse arguments, run the LLM pipeline, and save results."""
    parser = argparse.ArgumentParser(description="Parse arguments")
    parser.add_argument("--functions_definition", type=str,
                        default="data/input/functions_definition.json",
                        required=False, help="Parser filepath")
    parser.add_argument("--input", type=str,
                        default="data/input/function_calling_tests.json",
                        required=False, help="Input filepath")
    parser.add_argument("--output", type=str,
                        default="data/output/function_calling_results.json",
                        required=False, help="Output filepath")
    parser.add_argument("--verbose", action="store_true",
                        default=False, help="Print generation steps")
    args = parser.parse_args()

    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    except OSError as e:
        print(f"Invalid directory argument: {e}")
        sys.exit(1)

    func_defs = parse_func_defs(args.functions_definition)
    prompt_defs = parse_prompts(args.input)

    small_llm = SetupLLM("Qwen/Qwen3-0.6B", verbose=args.verbose)
    results: list[dict[str, Any]] = []
    for p in prompt_defs:
        result_dict: dict[str, Any] = {}
        result_dict["prompt"] = p.prompt
        res_func_name = small_llm.gen_func_name(p.prompt, func_defs)
        result_dict["name"] = res_func_name
        params: dict[str, Any] = {}
        for func in func_defs:
            if func.name == res_func_name:
                if func.parameters is not None:
                    for name, param in func.parameters.items():
                        raw = small_llm.gen_param_values(p.prompt, func, name,
                                                         param.type, params)
                        if (param.type not in ["number", "float",
                                               "int", "integer"]):
                            params[name] = raw
                        else:
                            try:
                                if param.type in ["number", "float"]:
                                    params[name] = float(raw)
                                elif param.type in ["int", "integer"]:
                                    params[name] = int(float(raw))
                            except ValueError:
                                params[name] = raw
        result_dict["parameters"] = params
        results.append(result_dict)
    gen_output_file(args.output, results)


if __name__ == "__main__":
    main()
