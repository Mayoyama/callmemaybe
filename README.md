*This project has been created as part of the 42 curriculum by speterse.*

# Call Me Maybe

> LLM function calling with constrained decoding using Qwen3-0.6B.

---

## Table of Contents

- [Description](#description)
- [Instructions](#instructions)

- [Algorithm](#algorithm)
- [Design Decisions](#design-decisions)
- [Performance Analysis](#performance-analysis)
- [Challenges Faced](#challenges-faced)
- [Testing Strategy](#testing-strategy)
- [Example Usage](#example-usage)
- [Bonus Features](#bonus-features)
  - [Comprehensive Test Suite](#comprehensive-test-suite)
  - [Support for Multiple LLM Models](#support-for-multiple-llm-models)
  - [Recoding the Tokenizer](#recoding-the-tokenizer)
  - [Advanced Error Recovery](#advanced-error-recovery)
  - [Performance Optimizations](#performance-optimizations)
  - [Visualization of the Generation Process](#visualization-of-the-generation-process)
  - [Support for Complex Nested Function Arguments](#support-for-complex-nested-function-arguments)
  - [Public Tokenizer Implementation](#public-tokenizer-implementation)
- [Resources & AI Usage](#resources)

---

## Description

Call Me Maybe is a Python 3 project that translates natural language user prompts into structured JSON function calls using a small language model (Qwen/Qwen3-0.6B). Rather than relying on fine-tuning or the model's built-in tool-use capabilities, it applies **constrained decoding** at inference time — restricting which tokens the model is allowed to generate based on the expected output structure. This keeps the model on task without any retraining.

The pipeline reads a set of function definitions and user prompts from JSON input files, selects the correct function for each prompt, extracts the required parameter values, and writes the results to a JSON output file.

<br/><a href="#table-of-contents">↑ Back to top</a>

---

## Instructions

### Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

### Install

```bash
make install
```

### Run

```bash
make run
```

Or with custom file paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Verbose

```bash
make verbose
```

Or equivalently:

```bash
uv run python -m src --verbose
```

### Lint

```bash
make lint
```

### Clean

```bash
make clean
```

<br/><a href="#table-of-contents">↑ Back to top</a>

---

## Algorithm

### Function Name Selection (`gen_func_name`)

1. Build a prompt listing all available functions with their descriptions and parameter types.
2. Encode the prompt into token IDs.
3. At each generation step, retrieve logits from the model.
4. Set the logit of any token to `-inf` if appending it to the current string would produce a value that is **not a valid prefix** of any available function name. This forces the model to stay on a valid decoding path.
5. Select the token with the highest remaining logit (greedy decoding).
6. Append it to the current function name string and repeat until an exact function name match is found.

### Parameter Value Extraction (`gen_param_values`)

1. Build a completion-style prompt that shows the function being called, the target parameter, and any already-extracted parameters as context lines.
2. End the prompt with `{param_name}: "` to prime the model to continue with the value.
3. Apply type-specific token constraints at each step:
   - **String**: block any token containing `"` (except the closing `"`), newline characters (`Ċ`), and tokens starting with `Ġ'` to prevent the model from escaping the string early.
   - **Number**: allow only tokens that keep the running value matching the regex `^-?\d*\.?\d*$`; stop on `}`, `,`, or `"`.
4. Decode the generated tokens back to a string, stripping the leading BPE space character (`Ġ`) and replacing internal `Ġ` with spaces.

<br/><a href="#table-of-contents">↑ Back to top</a>

---

## Design Decisions

### Constrained decoding over fine-tuning

Rather than training the model to produce structured output, logits are masked at inference time. This avoids any training cost and works with any model that exposes its logits, making the approach model-agnostic.

### Completion-style prompting for parameters

Parameters are extracted one at a time. Each prompt includes previously extracted parameters as `key: "value"` lines before the target parameter. This gives the model explicit context about what has already been found, which significantly reduces, but not completely removes, anchoring errors (e.g. a model generating `b=a` because it saw `a` last).

### Function name prompt iteration

Several prompt formulations were tested for the function name selection step:

| Prompt variant | Accuracy |
|---|---|
| `"user prompt: ...\navailable functions: {name}: {description}\nFunction name:"` | 82% |
| Same + parameter info added | 82% |
| Same + newlines between functions | 82% |
| Closing instruction changed to `"Based on the user request above, the function that should be called is:"` | ~100% on initial test set |

The final wording worked because the explicit back-reference to "the user request above" gave the model stronger context to anchor its choice, preventing it from defaulting to the statistically more familiar function (e.g. `fn_get_square_root` over `fn_greet`).

### BPE tokenisation handling

Qwen uses Byte-Pair Encoding where `Ġ` represents a leading space and `Ċ` represents a newline. Constraints and string cleaning are written to account for these, e.g. using `lstrip('Ġ')` rather than `strip('Ġ')` to preserve intentional leading spaces in extracted values.

### Separate prompts per parameter

Each parameter is extracted with its own LLM call rather than extracting all parameters in one pass. This keeps the constrained decoding logic simple and type-specific: the number constraint and string constraint are fundamentally different and would be difficult to apply simultaneously to a single generation pass.

<br/><a href="#table-of-contents">↑ Back to top</a>

---

## Performance Analysis

### Accuracy

| Test set | Functions | Prompts | Function Selection | Parameter Extraction | Combined Accuracy |
|---|---|---|---|---|---|
| School (original) | 5 | 12 | 12/12 | 10/12 | 91.7% |
| Extended (50 prompts) | 11 | 50 | 50/50 | 47/50 | 97% |
| Pytest suite | 21 | 100 | 97/100 | 90/100 | 93.5% |

I measured accuracy in terms of [number of prompts where the correct function was selected] plus [number of prompts where all parameter values were returned correctly] divided by [total number of prompts evaluated].

### Known failure cases

The remaining percentage of failures are attributable to the capabilities and limitations of the 0.6B model rather than the prompting or decoding logic:

- **Case transformation** — prompts like "make 'PYTHON' lowercase" cause the model to extract the full phrase rather than just the word, because a model this small lacks reliable instruction-following for transformation tasks.
- **Regex/replacement tasks** — the model struggles to correctly isolate regex patterns and replacement strings from natural language descriptions.
- **Negative numbers** — the model assigns higher probability to digit tokens than to `-`, so negative values are generated without their sign. This is a model prior issue, not a constraint issue.
- **B=A bias** - When two function names are too similar, such as `fn_celsius_to_fahrenheit` and `fn_fahrenheit_to_celsius`, the model can't reliably distinguish "A to B" vs "B to A" and will bias completely towards one over the other.
- **Limited Vocabulary base** - at 0.6B parameters, the model's semantic understanding is shallow enough that synonyms outside of its training distribution (eg "bigger", "higher") don't reliably map to the right description word ("larger").
- **Parameter ordering** - despite rephrasing the prompt both ways, the model ignores positional language like `prefix` and consistently extracts the more prominent word as `s1`, demonstrating that parameter order cannot be inferred from natural language position cues.

NB: While most of these could potentially be resolved with heuristics, I actively chose not to apply heuristics to the LLM prompts to push the model to its absolute limits.

<br/><a href="#table-of-contents">↑ Back to top</a>

---

## Challenges Faced

**Fine tuning of prompts** - It took a lot of time and 'trial & error' to achieve 90%+ accuracy across multiple param types. Often a small adjustment would improve the output of numbers but completely break strings, or would improve strings but then cause numeric function parameters to return the same value for both `a:` and `b:`.

**Understanding how to properly filter tokens** - not having had any experience with small LLMs, it was difficult to write and adjust the token filters correctly. At first certain prompts would loop infinitely without being able to generate a valid response.

**LLM unable to process negatives** - Regardless of how I adjusted the prompts I could not get Qwen to recognise that it had to consider negative values as negative without also adversely affecting other outputs. Since heuristics defeats the point of the project, I was unable to get Qwen to reliably handle these.

**Qwen accuracy issues** - Qwen also struggles to infer meaning from non-explicit prompts such as "Make 'x' upper case'. It would often think it needs to return the already processed answer rather than the param value, and I was ultimately unable to get 100% accuracy without heuristics.

**Blank line in completion prompt** — when no parameters had been extracted yet, joining an empty list produced a blank line before the target parameter name. This subtly changed the prompt structure and hurt accuracy. Fixed by only appending the extracted section when it is non-empty.

**Similar function names confusing the model** — functions with nearly identical descriptions (e.g. `convert_celsius_to_fahrenheit` vs `convert_fahrenheit_to_celsius`) cause the 0.6B model to select the wrong one. This is a known limitation of small models with limited world knowledge.

<br/><a href="#table-of-contents">↑ Back to top</a>

---

## Testing Strategy

Two additional test sets were used, both structured as a `functions_definition.json` (function schemas) paired with a `function_calling_tests.json` (natural language prompts). The pipeline output was compared manually against expected results, counting a result as correct only if both the function name and all parameter values matched.

**Test set 1** — 11 functions, 50 prompts. Used to iteratively develop and evaluate the prompting strategy. Each prompt change was re-run against the full set to confirm improvement or regression.

**Test set 2** — an extended set with new functions and prompts, used to verify that the final approach generalised beyond the original cases.

Edge cases specifically targeted during testing:
- **Similar function names** — e.g. `convert_celsius_to_fahrenheit` vs `convert_fahrenheit_to_celsius`, to check the constrained decoding could distinguish near-identical descriptions
- **Negative numbers** — to confirm the number constraint handled the `-` sign correctly
- **Type coercion** — ensuring `float` and `int` parameters were returned as the correct Python type, not a string
- **Parameter anchoring** — prompts where a second parameter value could be confused with the first (e.g. `b=a`), to verify whether adjustments to prompts helped
- **Functions with no parameters** — to confirm the pipeline handled `None` parameter sets without errors


<br/><a href="#table-of-contents">↑ Back to top</a>

---

## Example Usage

### Input — `functions_definition.json`

```json
[
  {
    "name": "fn_greet",
    "description": "Greet a person by name",
    "parameters": {
      "name": { "type": "string" }
    }
  }
]
```

### Input — `function_calling_tests.json`

```json
[
  { "prompt": "Say hello to Alice" }
]
```

### Output — `function_calling_results.json`

```json
[
  {
    "prompt": "Say hello to Alice",
    "name": "fn_greet",
    "parameters": {
      "name": "Alice"
    }
  }
]
```

<br/><a href="#table-of-contents">↑ Back to top</a>

---
## Bonus Features

### Comprehensive Test Suite

A pytest-based integration test suite (`test_pipeline.py`) was built to evaluate the LLM pipeline across a comprehensive set of functions and prompts. The suite tests the full pipeline end-to-end — function selection and parameter extraction — against known expected outputs.

#### Structure

The test suite covers 21 functions and 100 prompts, spanning all supported parameter types (`string`, `number`, `float`, `int`, `integer`). The model is loaded once per session using a module-scoped pytest fixture to avoid the overhead of reloading the model for each test.

Each test case is defined as a tuple of `(prompt, expected_function_name, expected_params)` and runs through the parametrize decorator, producing a separate pass/fail result per case.

#### Running the suite

```bash
uv run pytest
```

Example output:

```
platform win32 -- Python 3.14.4, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\python\callmemaybe
configfile: pyproject.toml
plugins: anyio-4.14.0
collected 100 items

test_pipeline.py .......xx............x......xx......................................x.......xx....xx................ 

90 passed, 10 xfailed in 153.39s (0:02:33)
```

### FAIL vs XFAIL

A **FAIL** is an unexpected failure — the pipeline produced an incorrect result where a correct one was expected. These indicate bugs or regressions that need to be investigated.

An **XFAIL** (expected failure) is a test that is known in advance to fail due to a documented limitation. It is marked with `pytest.mark.xfail` and reported separately. An xfailed test that fails is shown as `x` in the output — this is normal and expected. If an xfailed test *passes*, pytest reports it as `XPASS` (unexpected pass), which can indicate that a limitation has been resolved.

#### Why XFAIL matters for this project

The 0.6B model has several documented limitations that cannot be addressed through prompting or constraint logic alone. Rather than hiding these failures or removing the test cases, they are explicitly marked as xfail with a reason. This approach:

- Keeps the test suite honest — failures are documented, not omitted
- Makes the distinction clear between implementation bugs (FAIL) and model capability limits (XFAIL)
- Allows the suite to grow without known limitations polluting the pass rate
- Provides a record of exactly which prompt patterns the model cannot handle reliably

The 10 xfailed cases in this suite correspond directly to the known failure modes documented in the README performance analysis: direction-ambiguous function names, negative number extraction, regex synthesis, case transformation parameter isolation, and positional language inference.

<br/><a href="#table-of-contents">↑ Back to top</a>

---

### Support for Multiple LLM Models

The model identifier is passed as a parameter to `SetupLLM`, defaulting to `Qwen/Qwen3-0.6B`. Any compatible HuggingFace model can be substituted by passing a different model string at instantiation.

```bash
uv run python -m src  # uses Qwen/Qwen3-0.6B by default
```

<br/><a href="#table-of-contents">↑ Back to top</a>

---

### Recoding the Tokenizer

TODO — indicate whether `encode()` was replaced with a manual BPE implementation using `get_path_to_vocab_file` and `get_logits_from_input_ids` only.

<br/><a href="#table-of-contents">↑ Back to top</a>

---

### Advanced Error Recovery

TODO

<br/><a href="#table-of-contents">↑ Back to top</a>

---

### Performance Optimizations

TODO

<br/><a href="#table-of-contents">↑ Back to top</a>

---

### Visualization of the Generation Process

The `--verbose` flag prints the constrained decoding process live to the terminal
for every generation step. Run with:

```bash
uv run python -m src --verbose
```

Each line shows the label (function name or parameter being built), the string
constructed so far, the top 3 candidate tokens with their logit scores, and the
selected token:
```bash
fn [''] top: [('fn', 9.4766), ('f', 3.9551), ('!', -inf)] → 'fn'
fn ['fn'] top: [('_add', 23.125), ('_sub', 19.75), ('_g', 18.5938)] → '_add'
fn ['fn_add'] top: [('_numbers', 27.8281), ('number', 17.6094), ('', 16.375)] → '_numbers'
a [''] top: [('2', 25.1719), ('3', 18.2656), ('1', 17.3281)] → '2'
a ['2'] top: [('"', 19.2656), ('5', 15.0781), ('.', 25.0156)] → '.'
a ['2.'] top: [('5', 30.0469), ('3', 19.8906), ('4', 19.5625)] → '5'
a ['2.5'] top: [('"', 19.2656), ('5', 11.8984), ('0', 11.8828)] → '"'
```

The `!: -inf` token visible on the first step of every function selection is the
constraint actively blocking invalid tokens. The top 3 always reflect only
tokens the model genuinely prefers — the rest of the vocabulary has been masked
to `-inf` before this output is printed.

The output also makes known failure modes directly observable. For example, the
`replacement` parameter for "Replace all vowels in 'hello' with *" runs to its
50-token limit generating `asterisk*|*asterisk*|*...` in a loop, because the
model treats `*` as the word "asterisk" rather than a literal character. This is
one of the xfailed cases in the test suite and the visualization confirms exactly
why it fails.

<br/><a href="#table-of-contents">↑ Back to top</a>

---

### Support for Complex Nested Function Arguments

TODO

<br/><a href="#table-of-contents">↑ Back to top</a>

---

### Public Tokenizer Implementation

TODO — indicate whether public `encode` and `decode` methods were implemented and how they integrate with the constrained decoding pipeline.

<br/><a href="#table-of-contents">↑ Back to top</a>
___

## Resources

- [Python argparse documentation](https://docs.python.org/3/library/argparse.html)
- [TensorFlow Embedding Projector](https://projector.tensorflow.org/) 
- [Constrained Decoding explained (YouTube)](https://www.youtube.com/watch?v=Yad5fknpk2U)
- [Prompting and Prompt Engineering: A Comprehensive Guide](https://medium.com/@derrickryangiggs/prompting-and-prompt-engineering-a-comprehensive-guide-to-controlling-llm-behavior-9c8b417bd253)

### AI Usage

An AI assistant (Claude AI) was used as a reference and sounding board during development.

- Asking for clarification on how BPE tokenisation represents spaces and newlines (`Ġ`, `Ċ`) and what that meant for the token constraint logic
- Gaining understanding of the difference between how small and large LLMs process prompts
- Getting a second opinion on mypy errors and what the correct type annotations should be
- Using it to evaluate output accuracy across prompt iterations — pasting results and asking what percentage were correct and why certain cases failed
- Generating additional tests to confirm accuracy
- Explaining what pytest is and how to create a simple practice test
- Generating the skeleton and outline of the README

The core logic, prompt engineering iterations, and debugging were worked through independently.

<br/><a href="#table-of-contents">↑ Back to top</a>