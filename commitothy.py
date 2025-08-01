#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = "==3.12"
# dependencies = [
#     "openai",
# ]
# ///

import subprocess
import sys
import os
import argparse

from openai import OpenAI


def get_staged_diff():
    """Get the staged diff from git."""
    try:
        result = subprocess.run(
            ["git", "diff", "--staged"], capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError:
        print("Error: Not in a git repository or git command failed")
        sys.exit(1)


def get_changed_files():
    """Get list of staged files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--staged", "--name-only"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().split("\n") if result.stdout.strip() else []
    except subprocess.CalledProcessError:
        return []


def get_recent_messages(files, limit=20):
    """Get recent commit messages for the specified files."""
    if not files:
        return []

    try:
        separator = "===COMMIT_SEPARATOR==="
        cmd = [
            "git",
            "log",
            f"-{limit}",
            f"--pretty=format:%B{separator}",
            "--",
        ] + files
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        commits = result.stdout.strip().split(separator)
        return [commit for commit in commits if commit.strip()]
    except subprocess.CalledProcessError:
        return []


def llm_call(prompt, model="openrouter/auto", debug=False):
    """Call the OpenRouter API to generate a commit message."""
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY", ""),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that writes git commit messages.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=500,
            extra_headers={
                "HTTP-Referer": "https://mgalgs.io",
                "X-Title": "Commitothy",
            },
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        if debug:
            print(f"Error calling OpenRouter API: {e}")
        return


def generate_commit_message(
    diff, recent_messages, model="openrouter/auto", debug=False, num_retries=3
):
    """Generate a commit message using OpenAI."""
    examples_text = (
        "\n---\n".join(recent_messages) if recent_messages else "No examples available"
    )

    prompt = f"""You are a git commit message generator. Based on the git diff and examples of
recent commit messages for these files, generate an appropriate commit message.

Follow these rules strictly:
1. First line (summary) should be 50-72 characters max
2. If including a body, separate it from summary with a blank line
3. Wrap body lines at 72 characters
4. Use the same style, tone, and format as the examples
5. Focus on what changed and why, not how
6. Use imperative mood ("Fix", "Add", "Update", not "Fixed", "Added")
7. Pay close attention to the Summary line format in the examples. Example formats:
   - "subsystem1: [subsystem2:] Description" (Linux kernel style))
   - "<(fix|feat|chore)>[(subsystem)]: Description" (Conventional Commits style)
   - Something else that matches the examples
8. Only use ASCII, no Unicode characters

Git diff to analyze:
<diff>
{diff}
</diff>

Examples of recent commits that touched these files:
<examples>
{examples_text}
</examples>

Generate only the commit message with no other text."""

    if debug:
        print("Debug mode enabled. Prompt being sent to OpenRouter:")
        print(prompt)

    while num_retries > 0:
        response = llm_call(prompt, model=model, debug=debug)
        if response:
            return response.strip()
        num_retries -= 1


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate git commit messages using AI"
    )
    parser.add_argument(
        "--model",
        default="openrouter/auto",
        help="Model to use for generation (default: openrouter/auto)",
    )
    parser.add_argument(
        "--history-limit",
        default=20,
        type=int,
        help="How many recent commits to analyze",
    )
    parser.add_argument(
        "--num-retries",
        default=3,
        type=int,
        help="How many times to retry the LLM API call on failure",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it with: export OPENROUTER_API_KEY=your_api_key_here")
        sys.exit(1)

    diff = get_staged_diff()

    if not diff.strip():
        print("No staged changes found. Please stage some changes first.")
        sys.exit(1)

    files = get_changed_files()

    recent_messages = get_recent_messages(files, limit=args.history_limit)

    commit_message = generate_commit_message(
        diff,
        recent_messages,
        args.model,
        debug=args.debug,
        num_retries=args.num_retries,
    )

    print(commit_message)


if __name__ == "__main__":
    main()
