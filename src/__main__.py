import argparse
import os
from .parser import parse_func_defs, parse_prompts

def main():
    parser = argparse.ArgumentParser(description="Parse arguments")
    parser.add_argument("--functions_definition", type=str, default="data/input/functions_definition.json", required=False, help="Parser filepath")
    parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json", required=False, help="Input filepath")
    parser.add_argument("--output", type=str, default="data/output/function_calls.json", required=False, help="Output filepath")
    args = parser.parse_args()

    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
    except OSError as e:
        print(f"Invalid directory argument: {e}")
    
    func_defs = parse_func_defs(args.functions_definition)
    prompt_defs = parse_prompts(args.input)

if __name__ == "__main__":
    main()
