# Commitothy

**Commitothy** is a Python script that generates meaningful Git commit
messages using AI. It analyzes your staged changes and recent commit
history to produce well-formatted, context-aware commit messages in the
style of your existing project.

![screenshot](screenshot.png)

## Features

- ✅ Analyzes **staged Git diffs** automatically
- ✅ Learns from **recent commit messages** in your repo for style consistency
- ✅ Uses powerful LLMs via [OpenRouter](https://openrouter.ai) (support for multiple models)
- ✅ Enforces Git best practices:
  - Summary line ≤ 72 characters
  - Imperative mood ("Fix bug" not "Fixed bug")
  - Optional multi-line body with 72-character wrapping
  - Style matching (supports conventional commits, kernel style, etc.)
- 🔧 Configurable via command-line options
- 🐍 Standalone script using `uv` and `openai` - no heavy dependencies
- 💬 Adds a `Commit-Message-Co-Author` trailer indicating which AI model
  generated the message (disable with `--no-trailer`)
- 🧑‍⚖️ Optional AI **code review** appended as commented Markdown after the commit message
  (quick or full modes)

## Installation

1. Install [`uv`](https://github.com/astral-sh/uv) (recommended) or use any
   Python 3.12+ environment.
2. Clone or download `commitothy.py` and place it on your `PATH`:
   ```bash
   curl -o commitothy.py https://raw.githubusercontent.com/mgalgs/commitothy/main/commitothy.py
   chmod +x commitothy.py
   mv commitothy.py ~/bin/
   ```
3. Set your OpenRouter API key:
   ```bash
   export OPENROUTER_API_KEY=your_api_key_here
   ```
   Get a free API key at [OpenRouter.ai](https://openrouter.ai/keys).

## Usage

Stage your changes, then run:

```bash
commitothy.py
```

The default model is
[`openrouter/auto`](https://openrouter.ai/openrouter/auto). To use a
specific model (e.g. `google/gemini-2.5-flash`):

```bash
commitothy.py --model google/gemini-2.5-flash
```

Other options:
```bash
--history-limit N       # number of recent commits to analyze (default: 20)
--num-retries N         # retry failed API calls (default: 3)
--no-trailer            # don't add AI model attribution trailer
--debug                 # show full prompt sent to model
--code-review[=MODE]    # append an AI code review as commented Markdown after the
                        # commit message. MODE can be 'quick' or 'full'. If provided
                        # without a value, defaults to 'quick'. 'full' will include the
                        # full contents of all touched files for deeper context.
```

Use with `git commit`:
```bash
git commit -m "$(commitothy.py)"
```

Generate commit message with code review appended (quick):
```bash
commitothy.py --code-review
```

Generate commit message with code review appended (includes full file context for touched files):
```bash
commitothy.py --code-review=full
```

## How It Works

1. Collects the staged Git diff (`git diff --staged`) or `git show HEAD` when
   `--head`
2. Finds recently committed messages for the same files
3. Builds a smart prompt with code changes + style examples
4. Asks the LLM to generate a commit message matching the project's tone and format
5. Optionally generates a code review, using either just the diff (quick) or the diff
   plus the full contents of touched files (full)
6. Outputs clean message ready for `git commit`, with the review appended as `# `-prefixed comments

## Requirements

- [`uv`](https://github.com/astral-sh/uv)
- Git
- Shell environment with `OPENROUTER_API_KEY` set

Dependencies are managed inline via [script metadata](https://packaging.python.org/en/latest/specifications/inline-script-metadata/):
```toml
requires-python = "==3.12"
dependencies = [
    "openai",
]
```

## License

MIT

---

Made with ❤️ and `#!/usr/bin/env -S uv run --script`
