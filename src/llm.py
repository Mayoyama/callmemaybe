from llm_sdk import Small_LLM_Model  # type: ignore
from json import load, JSONDecodeError
from typing import Any
from src.parser import FuncDef
import sys
import re


class SetupLLM:
    """LLM wrapper for constrained function-call decoding using
    Qwen3-0.6B.
    """
    def __init__(self, llm_model: str = "Qwen/Qwen3-0.6B",
                 verbose: bool = False) -> None:
        """Initialise the LLM and load the BPE vocabulary.

        Args:
            llm_model: HuggingFace model identifier to load.
            verbose: a flag that will print logit operations to the screen 

        Raises:
            SystemExit: If the vocab file cannot be opened or parsed.
        """
        self.qwen_llm = Small_LLM_Model(llm_model)
        self.vocab_path = self.qwen_llm.get_path_to_vocab_file()
        self.verbose = verbose
        try:
            with open(self.vocab_path, encoding="utf-8") as file_obj:
                self.vocab_dict = load(file_obj)
        except (FileNotFoundError, PermissionError) as e:
            print(f"File error: {e}", file=sys.stderr)
            sys.exit(1)
        except JSONDecodeError as e:
            print(f"JSON error: {e}", file=sys.stderr)
            sys.exit(1)
        self.decode_dict = {
            token_id: token_string
            for token_string, token_id in self.vocab_dict.items()
        }

    def _print_step(self, label: str, current: str, logits: list[float],
                    selected: str) -> None:
        """Print the top 3 token candidates and selected token for a decoding step.

        Args:
            label: Prefix label (function name or parameter name).
            current: The string built so far in this generation pass.
            logits: Full logit list from the model.
            selected: The token that was selected.
        """
        top = sorted(
            ((i, s) for i, s in enumerate(logits) if i in self.decode_dict),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        top_tokens = [(self.decode_dict[i], round(s, 4)) for i, s in top]
        print(f"  {label} [{current!r}] top: {top_tokens} → {selected!r}")

    def gen_func_name(self, prompt: str, func_list: list[FuncDef]) -> str:
        """Select the best-matching function name for a user prompt.

        Uses constrained decoding to restrict token generation to
        names that are a valid prefix of an available function.

        Args:
            prompt: The natural language user request.
            func_list: List of available function definitions.

        Returns:
            The name of the selected function.
        """
        func_lines = []
        for func in func_list:
            if func.parameters is None:
                params = "none"
            else:
                params = ", ".join(
                    f"{name}: {param.type}"
                    for name, param in func.parameters.items()
                )
            func_lines.append(
                f"{func.name}: {func.description}"
                f" - parameters: {params}"
            )
        llm_prompt = (
            "user prompt: " + prompt
            + "\navailable functions:\n"
            + "\n".join(func_lines)
            + "\nBased on the user request above, "
            "the function that should be called is:"
        )
        encoded = self.qwen_llm.encode(llm_prompt)
        current_func_name = ""
        valid_names = [func.name for func in func_list]
        encoded_list = encoded[0].tolist()
        while (current_func_name not in valid_names):
            logits = self.qwen_llm.get_logits_from_input_ids(encoded_list)
            for token_string in self.vocab_dict.keys():
                if any(
                    name.startswith(current_func_name + token_string)
                    for name in valid_names
                ):
                    pass
                else:
                    token_id = self.vocab_dict[token_string]
                    logits[token_id] = float('-inf')
            max_idx = logits.index(max(logits))
            target_str = self.decode_dict[max_idx]
            if self.verbose:
                self._print_step("fn", current_func_name, logits, target_str)
            current_func_name += target_str
            encoded_list.append(max_idx)
        return current_func_name

    def gen_param_values(self, prompt: str, func_def: FuncDef, param_name: str,
                         param_type: str, param_dic: dict[str, Any]) -> str:
        """Extract a single parameter value from a user prompt.

        Uses constrained decoding to restrict tokens based on the
        expected parameter type (number or string).

        Args:
            prompt: The natural language user request.
            func_def: The function definition containing parameter metadata.
            param_name: The name of the parameter to extract.
            param_type: The expected type ('string', 'number', 'float', 'int').
            param_dic: Previously extracted parameters used as context.

        Returns:
            The extracted value as a string (caller handles numeric
            conversion).
        """
        extracted_lines = ("\n".join([f"{k}: \"{v}\"" for k,
                           v in param_dic.items()]))
        extracted_section = extracted_lines + "\n" if extracted_lines else ""

        llm_prompt = (
            f"Extract function call parameters from a user request.\n"
            f"Request: {prompt}\n"
            f"Function: {func_def.name} - {func_def.description}\n"
            f"Extract the input value for '{param_name}' ({param_type}) "
            f"from the Request.\n"
            f"{extracted_section}"
            f"{param_name}: \"")

        encoded = self.qwen_llm.encode(llm_prompt)
        encoded_list = encoded[0].tolist()
        current_value = ""
        if param_type in ("number", "float", "int", "integer"):
            i = 0
            regex_pattern = re.compile(r"^-?\d*\.?\d*$")
            while i <= 20:
                logits = self.qwen_llm.get_logits_from_input_ids(encoded_list)
                is_valid_number = any(char.isdigit() for char in current_value)
                for token_string in self.vocab_dict.keys():
                    if token_string in ('}', ',', '"') and is_valid_number:
                        continue
                    if regex_pattern.match(current_value + token_string):
                        continue
                    else:
                        token_id = self.vocab_dict[token_string]
                        logits[token_id] = float('-inf')
                max_idx = logits.index(max(logits))
                target_str = self.decode_dict[max_idx]
                if self.verbose:
                    self._print_step(param_name, current_value, logits, target_str)
                if target_str in ('}', ',', '"'):
                    break
                current_value += target_str
                encoded_list.append(max_idx)
                i += 1
        else:
            if not param_type == "string":
                print(f"Unrecognised parameter type {param_type}: Attempting "
                      "to process as a string", file=sys.stderr)
            i = 0
            while i <= 50:
                logits = self.qwen_llm.get_logits_from_input_ids(encoded_list)
                for token_string in self.vocab_dict.keys():
                    if (
                        ('"' in token_string and token_string != '"')
                        or 'Ċ' in token_string
                        or "Ġ'" in token_string
                    ):
                        token_id = self.vocab_dict[token_string]
                        logits[token_id] = float('-inf')
                max_idx = logits.index(max(logits))
                target_str = self.decode_dict[max_idx]
                if self.verbose:
                    self._print_step(param_name, current_value, logits, target_str)
                if target_str == '"':
                    break
                current_value += target_str
                encoded_list.append(max_idx)
                i += 1
        return current_value.lstrip('Ġ').replace('Ġ', ' ')
