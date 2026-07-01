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

#### List of additional (optional) flags:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \  # default: data/input/functions_definition.json
  --input data/input/function_calling_tests.json \               # default: data/input/function_calling_tests.json
  --output data/output/function_calling_results.json \           # default: data/output/function_calling_results.json
  --model "Qwen/Qwen3-0.6B" \                                    # default: Qwen/Qwen3-0.6B
  --bonus-encoder \                                              # use custom BPE tokenizer instead of SDK encoder
  --verbose                                                      # print constrained decoding steps live
```
#### Debug
 
```bash
make debug
```
 
Runs the program under Python's built-in debugger (`pdb`).
 
#### Pytest
 
```bash
make pytest
```
 
Runs the full integration test suite (100 prompts, 21 functions).
 
#### Run with 50-prompt set
 
```bash
make run-50
```
 
Runs the pipeline against the extended 50-prompt test set.

#### Run with nested test set
 
```bash
make run-nested
```

Runs the pipeline against the nested function arguments test set.

### Verbose

```bash
make verbose
```
Prints the constrained decoding process live to the terminal for every generation step.

#### Bonus Encoder
 
```bash
make bonus-encoder
```

Enables the custom BPE tokenizer instead of the SDK encoder. Not compatible with nested integer sub-fields.

### Lint

```bash
make lint
```

Runs flake8 and mypy with the required flags.

### Clean

```bash
make clean
```

Removes __pycache__, .mypy_cache, and compiled Python files.

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
| Pytest suite | 21 | 100 | 97/100 | 91/100 | 94% |

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
 
test_pipeline.py xxxxxxxxxX..........................................................................................

90 passed, 9 xfailed, 1 xpassed in 153.39s (0:02:33)
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

#### XFAIL cases in this suite

9 of the 10 marked cases xfail'ed as expected. The remaining case — one of the "Prefix" pair — xpasses with the SDK encoder, demonstrating that tokenisation choices can influence which edge cases the model handles correctly.

<br/><a href="#table-of-contents">↑ Back to top</a>

---

### Support for Multiple LLM Models

The `--model` flag allows any HuggingFace-compatible model to be substituted at runtime:

```bash
uv run python -m src  # uses Qwen/Qwen3-0.6B by default
uv run python -m src --model "Qwen/Qwen3-1.7B"
uv run python -m src --model "HuggingFaceTB/SmolLM2-1.7B"
```



#### Model Compatibility
 
| Model | Status | Error | Notes |
|---|---|---|---|
| Qwen/Qwen3-0.6B | ✅ Working | — | Default model. 94% accuracy (SDK), 92% (custom encoder). |
| Qwen/Qwen3-1.7B | ✅ Working | — | 86/100 passed. Introduces 4 new failures — over-interprets literal characters (`*` → `'asterisk'`) and misroutes "Append" prompts. |
| HuggingFaceTB/SmolLM2-1.7B | ⚠️ Poor | — | 47/100 passed. Collapses to routing nearly all prompts to `fn_greet`. Speed gain is an artefact of low-effort token selection. |
| HuggingFaceTB/SmolLM2-360M-Instruct | ⚠️ Poor | — | Routes almost everything to `fn_substitute_string_with_regex` regardless of prompt. Insufficient capacity to distinguish function descriptions. |
| microsoft/Phi-3-mini-4k-instruct | ❌ Failed to load | `rope_scaling` incompatibility | Incompatible `rope_scaling` configuration with the current version of `transformers`. |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | ❌ Failed to load | `UnicodeDecodeError` | Uses SentencePiece vocab (`.model`) instead of JSON (`vocab.json`). Incompatible with the vocab loading logic. |
| facebook/opt-125m | ❌ Failed to load | Vocab format error | OPT tokenizer format incompatible with the vocab loading logic. |
| Qwen/Qwen2-0.5B | ❌ Failed at runtime | `KeyError: 151643` | Larger vocabulary than Qwen3-0.6B — token IDs outside the loaded decode dictionary. Requires rebuilding vocab handling for Qwen2's tokenizer. |


#### Test Results

Three models were tested against the same 50-prompt set (11 functions). Results below are function selection accuracy on the 26 prompts with a clearly correct answer:
 
| Model | Family | Params | Function Selection |
|---|---|---|---|
| Qwen/Qwen3-0.6B | Qwen | 0.6B | 24/26 |
| Qwen/Qwen3-1.7B | Qwen | 1.7B | 25/26 |
| HuggingFaceTB/SmolLM2-1.7B | SmolLM | 1.7B | 17/26 |
 
Within the Qwen family, the larger model marginally outperforms the smaller one. SmolLM2-1.7B — despite matching Qwen3-1.7B in parameter count — performs significantly worse, failing all 6 greet prompts by routing them to `fn_add_numbers`. This demonstrates that model family and instruction-following training alignment matter more than raw parameter count for this task.

With Pytest:
| Model | Family | Params | Passed | Failed | XFailed | Time |
|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | Qwen | 0.6B | 90 | 0 | 10 | 153s |
| Qwen/Qwen3-1.7B | Qwen | 1.7B | 86 | 4 | 10 | 166s |
| HuggingFaceTB/SmolLM2-1.7B | SmolLM | 1.7B | 47 | 43 | 10 | 96s |
 
Qwen3-0.6B is the strongest performer despite being the smallest model. Qwen3-1.7B introduces 4 new failures not present in 0.6B — it over-interprets literal characters (`*` → `'star'`, `_` → `'x'`) and misroutes "Append" prompts, showing that more world knowledge can hurt on literal extraction tasks.
 
SmolLM2-1.7B is the fastest (~40% faster than either Qwen model) but collapses to routing nearly everything to `fn_greet`, producing 43 failures. The speed gain is an artefact of early, low-effort token selection rather than genuine efficiency.

<br/><a href="#table-of-contents">↑ Back to top</a>

---

### Recoding the Tokenizer

The SDK's `encode()` method was replaced with a custom BPE implementation that uses only `get_path_to_vocab_file()` and `get_logits_from_input_ids()` — the two permitted SDK primitives. This means the full pipeline from raw text to constrained token generation runs without any reliance on the SDK's tokeniser internals. The custom encoder is enabled with the `--bonus-encoder` flag; the default encoder uses the SDK's `encode()` for comparison.
 
See [Public Tokenizer Implementation](#public-tokenizer-implementation) for technical details and accuracy results.

<br/><a href="#table-of-contents">↑ Back to top</a>

---

### Advanced Error Recovery

Three failure modes are handled explicitly:

**Loop limit exceeded** — `gen_func_name` and `gen_param_values` both have iteration caps. If the generation loop hits the limit without resolving a valid result, a warning is printed to stderr with the prompt and iteration count. The cap in `gen_func_name` is **dynamic** — set to the length of the longest valid function name rather than an arbitrary constant.

**All tokens masked (`-inf` deadlock)** — if constrained decoding masks every token in the vocabulary, the model has no valid path forward. A dedicated guard detects this before calling `max()`, returning a safe fallback (`valid_names[0]` for function name, `'0'` for numeric parameters, `''` for string parameters) with a stderr warning.

**Deadlock-induced infinite loop** — without the loop limit, a `-inf` deadlock would cause silent infinite looping: the garbage token selected when all logits are `-inf` would likely trigger the same masking on the next iteration, repeating forever. The loop limit acts as a defensive backstop that caps the damage even if the deadlock guard were somehow missed.

All error messages follow the format `[function_name()]: description`, printed to `sys.stderr`.

<br/><a href="#table-of-contents">↑ Back to top</a>

---

### Performance Optimizations

**Caching** — `get_logits_from_input_ids` is called once per token per generation step. Since the model is deterministic, calling it twice with the same token sequence returns identical logits. A dictionary cache (`self.cache: dict[tuple[int, ...], list[float]]`) stores results keyed by the full token sequence as a tuple. The cache is cleared at the start of each `gen_func_name` and `gen_param_values` call to prevent stale logits from bleeding across prompts. On the pytest suite (100 prompts), this reduced runtime from ~153s to ~146s — approximately a 5% reduction. The cached list is always copied before mutation to prevent corrupting stored results.

A persistent cross-prompt cache (no clearing between calls) was also tested but provided no benefit — each prompt produces a unique token sequence from the first token, so the cache never hits across calls. A vocab pre-filtering optimisation was also attempted, using the set of characters appearing in valid function names to skip irrelevant tokens before the prefix check. This broke function selection because BPE tokens include the `Ġ` leading-space character, which is absent from function names but required for correct tokenisation. The pre-filter was removed.

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

A nested JSON object parameter is one where the value is itself a JSON object — a set of named sub-fields each with their own type, rather than a flat scalar like `"name": "Alice"`. For example, an `address` parameter might contain `city`, `postcode`, and `street` as separate typed fields inside a single object.

To support this, `ParameterType` in `parser.py` was extended with an optional `properties` field (`dict[str, 'ParameterType'] | None`), using `model_rebuild()` to resolve the self-referential type annotation at class definition time. A standalone helper function `gen_object_param()` was added to `__main__.py`; it iterates over the sub-fields and calls `gen_param_values()` once per sub-field, applying the same type coercion logic (`number`, `float`, `int`, `integer`) as top-level parameters. In the main extraction loop, any parameter with `type == "object"` branches to `gen_object_param()` and skips the standard scalar extraction path via `continue`.

**Note:** the bonus encoder (`--bonus-encoder`) is not compatible with nested integer sub-fields — use the default SDK encoder when running nested tests.

#### Creating a Nested Object Test

A nested function definition uses `"type": "object"` with a `"properties"` sub-object listing each field and its type:

```json
[
  {
    "name": "fn_example",
    "description": "Description of what the function does.",
    "parameters": {
      "param_name": {
        "type": "object",
        "properties": {
          "field_one": { "type": "string" },
          "field_two": { "type": "integer" }
        }
      }
    },
    "returns": { "type": "object" }
  }
]
```

The corresponding prompts file is unchanged from the flat format:

```json
[
  { "prompt": "Your natural language prompt here" }
]
```
 

<br/><a href="#table-of-contents">↑ Back to top</a>

---

### Public Tokenizer Implementation

Byte Pair Encoding (BPE) is a tokenisation algorithm that starts with individual characters and repeatedly merges the most frequently co-occurring pairs into a single token. This produces a vocabulary of subword units — common words become single tokens, while rare words get split into smaller recognisable pieces.

A custom BPE encoder was implemented in `src/tokenizer.py` as an alternative to the `encode()` method provided by `llm_sdk`. It can be enabled with the `--bonus-encoder` flag and uses only `get_path_to_vocab_file()` and `get_logits_from_input_ids()` from the SDK.
 
The encoder works in two steps. First, `raw_to_BPE()` converts the input string into BPE space by encoding it to UTF-8 bytes and mapping each byte through a `bytes_to_unicode()` table — the same 256-entry mapping used by all GPT-2-derived tokenisers. This correctly handles non-ASCII characters: the degree symbol `°` (U+00B0), for example, becomes two BPE characters `Â°` rather than a single incorrect lookup. Space and newline are handled automatically by the table (`0x20` → `Ġ`, `0x0A` → `Ċ`). Second, `encode()` performs a greedy longest-match scan over the BPE string, looking up each substring in the vocabulary dict and appending the corresponding token ID.
 
The token IDs produced by `encode()` are passed directly to `get_logits_from_input_ids()` at each constrained decoding step, and each selected token ID is appended to the sequence before the next step — demonstrating the full encode → logits → constrained selection loop without any SDK tokeniser involvement.
 
A public `decode()` method was not implemented because it was not needed: parameter values are reconstructed character by character during generation rather than by decoding a finished token sequence. The only post-processing required is stripping the leading BPE space character (`Ġ`) and replacing internal `Ġ` with spaces, which is handled inline.
 
Using the pytest testing suite, the custom encoder passes 87/100 tests compared to 90/100 for the SDK encoder. The 3 remaining failures are model capability issues unrelated to tokenisation. An initial implementation that skipped the `bytes_to_unicode` step (replacing only space and newline) produced 86/100 — the missing case being prompts containing `°`, which was tokenised as the wrong token ID.

#### Accuracy

| Test set | Functions | Prompts | Function Selection | Parameter Extraction | Combined Accuracy |
|---|---|---|---|---|---|
|SDK encoder | 21 | 100 | 97/100 | 91/100 | 94% |
|Custom encoder | 21 | 100 | 96/100 | 88/100 | 92% |

An xpass (unexpected pass) is a test that was marked as an expected failure but passed anyway, signalling that the model performed better than anticipated. In this case, the two encoders xpass opposite prompts in the "Prefix" pair, demonstrating that tokenisation choices can measurably influence model behaviour even when the underlying prompt is identical.

The pair in question is "Prefix 'bothered' with 'un'" and "Prefix 'un' with 'bothered'" — two prompts that are semantically equivalent but phrased differently, where the SDK encoder passes one and the custom encoder passes the other.

<br/><a href="#table-of-contents">↑ Back to top</a>
___

## Resources

- [Python argparse documentation](https://docs.python.org/3/library/argparse.html)
- [TensorFlow Embedding Projector](https://projector.tensorflow.org/) 
- [Constrained Decoding explained (YouTube)](https://www.youtube.com/watch?v=Yad5fknpk2U)
- [Prompting and Prompt Engineering: A Comprehensive Guide](https://medium.com/@derrickryangiggs/prompting-and-prompt-engineering-a-comprehensive-guide-to-controlling-llm-behavior-9c8b417bd253)
- [ASCII Table with All 256 Character codes in decimal, hexadecimal, octal and binary](https://www.sciencebuddies.org/science-fair-projects/references/ascii-table?__cf_chl_f_tk=Aw4II72VGTkWsg1br6e8whJ3zlWRs.B34QKmIaH3JtE-1782860051-1.0.1.1-zUHs16oMCyamzm.ZPiguMnNYjz.S3nlAU1.9PlUft3w)
- [What is UTF-8](https://blog.hubspot.com/website/what-is-utf-8)
- [What is Tokenization?](https://www.geeksforgeeks.org/nlp/what-is-tokenization/)
- [Byte-Pair Encoding (BPE) in NLP](https://www.geeksforgeeks.org/nlp/byte-pair-encoding-bpe-in-nlp/)

### AI Usage

An AI assistant (Claude AI) was used as a reference and sounding board during development.

- Asking for clarification on how BPE tokenisation represents spaces and newlines (`Ġ`, `Ċ`) and what that meant for the token constraint logic
- Gaining understanding of the difference between how small and large LLMs process prompts
- Getting a second opinion on mypy errors and what the correct type annotations should be
- Using it to evaluate output accuracy across prompt iterations — pasting results and asking what percentage were correct and why certain cases failed
- Generating additional tests to confirm accuracy
- Explaining what pytest is and how to create a simple practice test
- Generating the skeleton and outline of the README.md
- Grammar, spelling and formatting inconsistencies in the README.md

The core logic, prompt engineering iterations, and debugging were worked through independently.

<br/><a href="#table-of-contents">↑ Back to top</a>