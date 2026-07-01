import pytest
from src.llm import SetupLLM
from src.parser import FuncDef, ParameterType
from typing import Any


@pytest.fixture(scope="module")
def llm(bonus_encoder: Any) -> SetupLLM:
    """Load the LLM once for the entire test session."""
    return SetupLLM("Qwen/Qwen3-0.6B", bonus_encoder=bonus_encoder)


FUNC_DEFS = [
    FuncDef(
        name="fn_greet",
        description="Greet a person by name",
        parameters={"name": ParameterType(type="string")}
    ),
    FuncDef(
        name="fn_to_lowercase",
        description="Convert a string to lowercase and return the result.",
        parameters={"s": ParameterType(type="string")}
    ),
    FuncDef(
        name="fn_substitute_string_with_regex",
        description=(
            "Replace all matches of a regex pattern"
            " in a string with a replacement."
        ),
        parameters={
            "source_string": ParameterType(type="string"),
            "regex":         ParameterType(type="string"),
            "replacement":   ParameterType(type="string"),
        }
    ),
    FuncDef(
        name="fn_divide_numbers",
        description=(
            "Divide the first number by the second and return the result."
        ),
        parameters={
            "a": ParameterType(type="number"),
            "b": ParameterType(type="number"),
        }
    ),
    FuncDef(
        name="fn_modulo",
        description=(
            "Return the remainder when the first number"
            " is divided by the second."
        ),
        parameters={
            "a": ParameterType(type="number"),
            "b": ParameterType(type="number"),
        }
    ),
    FuncDef(
        name="fn_multiply_numbers",
        description="Multiply two numbers together and return the result.",
        parameters={
            "a": ParameterType(type="number"),
            "b": ParameterType(type="number"),
        }
    ),
    FuncDef(
        name="fn_max_of_two",
        description="Return the larger of two numbers.",
        parameters={
            "a": ParameterType(type="number"),
            "b": ParameterType(type="number"),
        }
    ),
    FuncDef(
        name="fn_pad_string",
        description=(
            "Pad a string to a given width with a fill character"
            " and return the result."
        ),
        parameters={
            "s":     ParameterType(type="string"),
            "width": ParameterType(type="integer"),
            "char":  ParameterType(type="string"),
        }
    ),
    FuncDef(
        name="fn_celsius_to_fahrenheit",
        description="Convert a temperature from Celsius to Fahrenheit.",
        parameters={"celsius": ParameterType(type="number")}
    ),
    FuncDef(
        name="fn_fahrenheit_to_celsius",
        description="Convert a temperature from Fahrenheit to Celsius.",
        parameters={"fahrenheit": ParameterType(type="number")}
    ),
    FuncDef(
        name="fn_concatenate_strings",
        description="Concatenate two strings and return the result.",
        parameters={
            "s1": ParameterType(type="string"),
            "s2": ParameterType(type="string"),
        }
    ),
    FuncDef(
        name="fn_repeat_string",
        description=(
            "Repeat a string a given number of times and return the result."
        ),
        parameters={
            "s": ParameterType(type="string"),
            "n": ParameterType(type="number"),
        }
    ),
    FuncDef(
        name="fn_count_words",
        description="Count the number of words in a string.",
        parameters={"s": ParameterType(type="string")}
    ),
    FuncDef(
        name="fn_add_numbers",
        description="Add two numbers together",
        parameters={
            "a": ParameterType(type="float"),
            "b": ParameterType(type="float"),
        }
    ),
    FuncDef(
        name="fn_to_uppercase",
        description="Convert a string to uppercase and return the result.",
        parameters={"s": ParameterType(type="string")}
    ),
    FuncDef(
        name="fn_average_of_two",
        description="Return the average of two numbers.",
        parameters={
            "a": ParameterType(type="number"),
            "b": ParameterType(type="number"),
        }
    ),
    FuncDef(
        name="fn_factorial",
        description="Return the factorial of a non-negative integer.",
        parameters={"n": ParameterType(type="integer")}
    ),
    FuncDef(
        name="fn_power",
        description="Raise a number to a given power and return the result.",
        parameters={
            "base":     ParameterType(type="number"),
            "exponent": ParameterType(type="int"),
        }
    ),
    FuncDef(
        name="fn_string_length",
        description="Return the number of characters in a string.",
        parameters={"s": ParameterType(type="string")}
    ),
    FuncDef(
        name="fn_truncate_string",
        description=(
            "Truncate a string to a given number of characters"
            " and return the result."
        ),
        parameters={
            "s":      ParameterType(type="string"),
            "length": ParameterType(type="integer"),
        }
    ),
    FuncDef(
        name="fn_reverse_string",
        description="Reverse a string",
        parameters={"s": ParameterType(type="string")}
    ),
]

CASES = [
    # -- known model limitations --
    pytest.param(
        "What is the sum of -5 and 6?",
        "fn_add_numbers",
        {"a": -5.0, "b": 6.0},
        marks=pytest.mark.xfail(
            reason=(
                "Known error: negatives do not work"
                " without heuristics"
            ),
        ),
    ),
    pytest.param(
        "Add -8.1 and 7.2",
        "fn_add_numbers",
        {"a": -8.1, "b": 7.2},
        marks=pytest.mark.xfail(
            reason=(
                "Known error: negatives do not work"
                " without heuristics"
            ),
        ),
    ),
    pytest.param(
        "Make 'PYTHON' lowercase",
        "fn_to_lowercase",
        {"s": "PYTHON"},
        marks=pytest.mark.xfail(
            reason=(
                "Known error: make x upper/lower"
                " fails to extract the word"
            ),
        ),
    ),
    pytest.param(
        "Replace all numbers in 'Hello 34 World' with X",
        "fn_substitute_string_with_regex",
        {
            "source_string": "Hello 34 World",
            "regex": r"\d+",
            "replacement": "X",
        },
        marks=pytest.mark.xfail(
            reason="Known error: model struggles to infer regex rules",
        ),
    ),
    pytest.param(
        "Replace all vowels in 'hello' with *",
        "fn_substitute_string_with_regex",
        {
            "source_string": "hello",
            "regex": "[aeiou]",
            "replacement": "*",
        },
        marks=pytest.mark.xfail(
            reason="Known error: model struggles to infer regex rules",
        ),
    ),
    pytest.param(
        "Which has a higher numeric value, 100 or 99?",
        "fn_max_of_two",
        {"a": 100.0, "b": 99.0},
        marks=pytest.mark.xfail(
            reason=(
                "Known error: limited vocabulary"
                " makes inferring synonyms difficult"
            ),
        ),
    ),
    pytest.param(
        "212 degrees Fahrenheit in Celsius?",
        "fn_fahrenheit_to_celsius",
        {"fahrenheit": 212.0},
        marks=pytest.mark.xfail(
            reason="Known error: F→C name too similar to C→F",
        ),
    ),
    pytest.param(
        "Convert 72 Fahrenheit to Celsius",
        "fn_fahrenheit_to_celsius",
        {"fahrenheit": 72.0},
        marks=pytest.mark.xfail(
            reason="Known error: F→C name too similar to C→F",
        ),
    ),
    pytest.param(
        "Prefix 'bothered' with 'un' to make a single string",
        "fn_concatenate_strings",
        {"s1": "un", "s2": "bothered"},
        marks=pytest.mark.xfail(
            reason=(
                "Known error: 0.6B unable to infer"
                " positional meaning of 'prefix'"
            ),
        ),
    ),
    pytest.param(
        "Prefix 'un' with 'bothered' to make a single string",
        "fn_concatenate_strings",
        {"s1": "bothered", "s2": "un"},
        marks=pytest.mark.xfail(
            reason=(
                "Known error: 0.6B unable to infer"
                " positional meaning of 'prefix'"
            ),
        ),
    ),

    # -- standard cases --
    ("Greet Alice", "fn_greet", {"name": "Alice"}),
    ("Greet Shane", "fn_greet", {"name": "Shane"}),
    ("Say hello to Lucas", "fn_greet", {"name": "Lucas"}),
    ("Salute Mariana", "fn_greet", {"name": "Mariana"}),
    ("Welcome Thibault", "fn_greet", {"name": "Thibault"}),
    ("What is the sum of 2 and 3?", "fn_add_numbers", {"a": 2.0, "b": 3.0}),
    (
        "What is the sum of forty-two and thirteen?",
        "fn_add_numbers",
        {"a": 42.0, "b": 13.0},
    ),
    ("Add 8.1 and 7.2", "fn_add_numbers", {"a": 8.1, "b": 7.2}),
    ("109 + 652 =", "fn_add_numbers", {"a": 109.0, "b": 652.0}),
    ("five plus nine equals?", "fn_add_numbers", {"a": 5.0, "b": 9.0}),
    ("Reverse the string 'hello'", "fn_reverse_string", {"s": "hello"}),
    (
        "Write 'Final Fantasy' in reverse order?",
        "fn_reverse_string",
        {"s": "Final Fantasy"},
    ),
    (
        "The string 'backup' backwards is?",
        "fn_reverse_string",
        {"s": "backup"},
    ),
    (
        "The string 'forwards' backwards is?",
        "fn_reverse_string",
        {"s": "forwards"},
    ),
    (
        "What is 2 to the power of 8?",
        "fn_power",
        {"base": 2.0, "exponent": 8},
    ),
    (
        "Calculate 3 raised to the power of 4",
        "fn_power",
        {"base": 3.0, "exponent": 4},
    ),
    ("What is 5^3?", "fn_power", {"base": 5.0, "exponent": 3}),
    (
        "Raise 10 to the power of 6",
        "fn_power",
        {"base": 10.0, "exponent": 6},
    ),
    (
        "What is 2.5 to the power of 2?",
        "fn_power",
        {"base": 2.5, "exponent": 2},
    ),
    (
        'Convert "HELLO WORLD" to lowercase',
        "fn_to_lowercase",
        {"s": "HELLO WORLD"},
    ),
    (
        "What is 'PROGRAMMING' in lowercase?",
        "fn_to_lowercase",
        {"s": "PROGRAMMING"},
    ),
    (
        "Pad the string 'hello' to width 10 with character '*'",
        "fn_pad_string",
        {"s": "hello", "width": 10, "char": "*"},
    ),
    (
        "Pad the string 'abc' to width 5 with character '0'",
        "fn_pad_string",
        {"s": "abc", "width": 5, "char": "0"},
    ),
    (
        "Pad the string 'hey' to width 8 with character 'D'",
        "fn_pad_string",
        {"s": "hey", "width": 8, "char": "D"},
    ),
    (
        "Pad the string 'test' to width 6 with character '_'",
        "fn_pad_string",
        {"s": "test", "width": 6, "char": "_"},
    ),
    (
        "Substitute 'cat' with 'dog' in 'the cat sat'",
        "fn_substitute_string_with_regex",
        {
            "source_string": "the cat sat",
            "regex": "cat",
            "replacement": "dog",
        },
    ),
    (
        "How long is the string 'hello'?",
        "fn_string_length",
        {"s": "hello"},
    ),
    (
        "What is the length of 'Python'?",
        "fn_string_length",
        {"s": "Python"},
    ),
    (
        "Tell me the total number of characters in 'program'",
        "fn_string_length",
        {"s": "program"},
    ),
    (
        "How many characters are in 'test'?",
        "fn_string_length",
        {"s": "test"},
    ),
    (
        "What is the average of 10 and 20?",
        "fn_average_of_two",
        {"a": 10.0, "b": 20.0},
    ),
    (
        "What is the average of 3.5 and 4.5?",
        "fn_average_of_two",
        {"a": 3.5, "b": 4.5},
    ),
    (
        "Calculate the average of 100 and 200",
        "fn_average_of_two",
        {"a": 100.0, "b": 200.0},
    ),
    (
        "What is the average of 7 and 3?",
        "fn_average_of_two",
        {"a": 7.0, "b": 3.0},
    ),
    (
        "Find the average of 15 and 25",
        "fn_average_of_two",
        {"a": 15.0, "b": 25.0},
    ),
    ("What is the factorial of 5?", "fn_factorial", {"n": 5}),
    ("Calculate 7 factorial", "fn_factorial", {"n": 7}),
    ("What is 0 factorial?", "fn_factorial", {"n": 0}),
    ("Find 10 factorial", "fn_factorial", {"n": 10}),
    ("What is 3 factorial?", "fn_factorial", {"n": 3}),
    ("Convert 'hello' to uppercase", "fn_to_uppercase", {"s": "hello"}),
    ("Make 'python' all caps", "fn_to_uppercase", {"s": "python"}),
    ("What is 'world' in uppercase?", "fn_to_uppercase", {"s": "world"}),
    (
        "Convert 'programming' to uppercase letters",
        "fn_to_uppercase",
        {"s": "programming"},
    ),
    ("Multiply 6 by 7", "fn_multiply_numbers", {"a": 6.0, "b": 7.0}),
    ("Multiply 6 and 7", "fn_multiply_numbers", {"a": 6.0, "b": 7.0}),
    ("What is 12 times 3?", "fn_multiply_numbers", {"a": 12.0, "b": 3.0}),
    (
        "What is the product of 8 and 9?",
        "fn_multiply_numbers",
        {"a": 8.0, "b": 9.0},
    ),
    (
        "Calculate 2.5 multiplied by 4",
        "fn_multiply_numbers",
        {"a": 2.5, "b": 4.0},
    ),
    (
        "Find the product of 13 and 11",
        "fn_multiply_numbers",
        {"a": 13.0, "b": 11.0},
    ),
    ("69*69=?", "fn_multiply_numbers", {"a": 69.0, "b": 69.0}),
    ("69 x 69 = ?", "fn_multiply_numbers", {"a": 69.0, "b": 69.0}),
    ("Divide 10 by 2", "fn_divide_numbers", {"a": 10.0, "b": 2.0}),
    (
        "What is 100 divided by 4?",
        "fn_divide_numbers",
        {"a": 100.0, "b": 4.0},
    ),
    ("Divide 7.5 by 2.5", "fn_divide_numbers", {"a": 7.5, "b": 2.5}),
    ("What is 1 divided by 3?", "fn_divide_numbers", {"a": 1.0, "b": 3.0}),
    ("Divide 50 by 10", "fn_divide_numbers", {"a": 50.0, "b": 10.0}),
    ("9 / 3 is?", "fn_divide_numbers", {"a": 9.0, "b": 3.0}),

    ("What is 10 modulo 3?", "fn_modulo", {"a": 10.0, "b": 3.0}),
    ("Find the remainder of 17%5", "fn_modulo", {"a": 17.0, "b": 5.0}),
    ("What is 100 mod 7?", "fn_modulo", {"a": 100.0, "b": 7.0}),
    ("Modulo of 256 and 16", "fn_modulo", {"a": 256.0, "b": 16.0}),
    (
        "What is the larger of 3 and 7?",
        "fn_max_of_two",
        {"a": 3.0, "b": 7.0},
    ),
    (
        "Return the max of 4.5 and 4.6",
        "fn_max_of_two",
        {"a": 4.5, "b": 4.6},
    ),
    (
        "What is the maximum of 0 and 1?",
        "fn_max_of_two",
        {"a": 0.0, "b": 1.0},
    ),
    (
        "Find the larger number between 50 and 25",
        "fn_max_of_two",
        {"a": 50.0, "b": 25.0},
    ),
    (
        "Convert 0 degrees Celsius to Fahrenheit",
        "fn_celsius_to_fahrenheit",
        {"celsius": 0.0},
    ),
    (
        "What is 100 Celsius in Fahrenheit?",
        "fn_celsius_to_fahrenheit",
        {"celsius": 100.0},
    ),
    ("Convert 42 °C to °F", "fn_celsius_to_fahrenheit", {"celsius": 42.0}),
    (
        "Convert 25 Celsius to Fahrenheit",
        "fn_celsius_to_fahrenheit",
        {"celsius": 25.0},
    ),
    (
        "Concatenate 'hello' and ' world'",
        "fn_concatenate_strings",
        {"s1": "hello", "s2": "world"},
    ),
    (
        "Join 'foo' and 'bar' together",
        "fn_concatenate_strings",
        {"s1": "foo", "s2": "bar"},
    ),
    (
        "Combine the strings 'Python' and ' is fun'",
        "fn_concatenate_strings",
        {"s1": "Python", "s2": "is fun"},
    ),
    (
        "Append 'ness' to 'kind'",
        "fn_concatenate_strings",
        {"s1": "kind", "s2": "ness"},
    ),
    (
        "Combine 'intense' & 'ly' into one string",
        "fn_concatenate_strings",
        {"s1": "intense", "s2": "ly"},
    ),
    (
        "Repeat the string 'ha' 3 times",
        "fn_repeat_string",
        {"s": "ha", "n": 3.0},
    ),
    ("Repeat 'abc' 5 times", "fn_repeat_string", {"s": "abc", "n": 5.0}),
    (
        "Repeat the word 'hello' 2 times",
        "fn_repeat_string",
        {"s": "hello", "n": 2.0},
    ),
    ("Repeat 'na' 4 times", "fn_repeat_string", {"s": "na", "n": 4.0}),
    (
        "Truncate 'hello world' to 5 characters",
        "fn_truncate_string",
        {"s": "hello world", "length": 5},
    ),
    (
        "Truncate 'Python programming' to 6 characters",
        "fn_truncate_string",
        {"s": "Python programming", "length": 6},
    ),
    (
        "Truncate 'abcdefgh' to 3 characters",
        "fn_truncate_string",
        {"s": "abcdefgh", "length": 3},
    ),
    (
        "Truncate 'good morning' to 4 characters",
        "fn_truncate_string",
        {"s": "good morning", "length": 4},
    ),
    (
        "Truncate 'testing' to 4 characters",
        "fn_truncate_string",
        {"s": "testing", "length": 4},
    ),
    (
        "Truncate 'hello' to 2 characters",
        "fn_truncate_string",
        {"s": "hello", "length": 2},
    ),
    (
        "How many words are in 'the quick brown fox'?",
        "fn_count_words",
        {"s": "the quick brown fox"},
    ),
    (
        "Count the words in 'hello world'",
        "fn_count_words",
        {"s": "hello world"},
    ),
    (
        "How many words does 'Python is a great language' have?",
        "fn_count_words",
        {"s": "Python is a great language"},
    ),
    (
        "Count words in 'one two three four five'",
        "fn_count_words",
        {"s": "one two three four five"},
    ),
    (
        "How many words are in 'just one'?",
        "fn_count_words",
        {"s": "just one"},
    ),
]


@pytest.mark.parametrize("prompt,expected_name,expected_params", CASES)
def test_function_call(
    llm: SetupLLM,
    prompt: str,
    expected_name: str,
    expected_params: dict,
) -> None:
    """Test that the pipeline selects the correct function and extracts
    parameters.
    """
    # if prompt in XFAIL_PROMPTS:
    #     pytest.xfail("Model biases — known 0.6B limitation")
    result_name = llm.gen_func_name(prompt, FUNC_DEFS)
    assert result_name == expected_name

    func = next(f for f in FUNC_DEFS if f.name == result_name)
    params: dict = {}
    if func.parameters is not None:
        for param_name, param in func.parameters.items():
            raw = llm.gen_param_values(
                prompt, func, param_name, param.type, params
            )
            if param.type in ("number", "float"):
                params[param_name] = float(raw)
            elif param.type in ("int", "integer"):
                params[param_name] = int(float(raw))
            else:
                params[param_name] = raw

    assert params == expected_params
