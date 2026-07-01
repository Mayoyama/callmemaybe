import argparse
import os
import sys
from src.parser import parse_func_defs, parse_prompts
from src.llm import SetupLLM
from src.utils import gen_output_file, recursive_gen_object_param
from typing import Any


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
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B",
                        required=False, help="Option to change LLM being used")
    parser.add_argument("--verbose", action="store_true",
                        default=False, help="Print generation steps")
    parser.add_argument("--bonus-encoder", action="store_true",
                        default=False,
                        help="Use custom BPE tokenizer "
                        "(encode/decode) instead of llm_sdk defaults")
    args = parser.parse_args()

    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    except OSError as e:
        print(f"Invalid directory argument: {e}")
        sys.exit(1)

    func_defs = parse_func_defs(args.functions_definition)
    prompt_defs = parse_prompts(args.input)

    small_llm = SetupLLM(args.model, verbose=args.verbose,
                         bonus_encoder=args.bonus_encoder)
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
                        if param.type == 'object':
                            props = param.properties
                            if props is not None:
                                params[name] = (
                                    recursive_gen_object_param(p.prompt, func,
                                                               props, params,
                                                               small_llm))
                            continue

                        raw = small_llm.gen_param_values(p.prompt, func, name,
                                                         param.type, params)
                        if (param.type not in ["number", "float",
                                               "int", "integer"]):
                            params[name] = raw
                        else:
                            try:
                                if param.type in ["number", "float"]:
                                    value = float(raw)
                                    params[name] = (
                                        int(value) if value.is_integer()
                                        else value
                                    )
                                elif param.type in ["int", "integer"]:
                                    params[name] = int(float(raw))
                            except ValueError:
                                params[name] = raw
        result_dict["parameters"] = params
        results.append(result_dict)
    gen_output_file(args.output, results)


if __name__ == "__main__":
    main()
