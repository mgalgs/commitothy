#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = "==3.12"
# dependencies = [
#     "openai",
# ]
# ///

from pathlib import Path
import argparse
import os
import re
import subprocess
import sys

from openai import OpenAI


PROMPT_TEMPLATE = """You are a commit message generator. Based on the context below (git diff, examples of recent commit messages for these files, etc), generate an appropriate commit message.

{preamble}

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

{extra_context}

Generate only the commit message with no other text.

Remember that you are describing the rationale behind the change, not the
implementation details. Explain *why* the change was made, not *how* it was
done.

IMPORANT: Imitate the style and format of the examples as closely as possible.
Pay attention to the lengths of messages, tone, personality, and structure.

REMEMBER: do not add trailer lines. For example, `Signed-off-by:`, `Commit-Message-Co-Author:`,
`Change-Id:`, and ANY OTHER trailer should NOT be included.

{postamble}
"""


def git(args):
    try:
        result = subprocess.run(
            ["git"] + args, capture_output=True, text=True, check=True
        )
        return result.stdout
    except subprocess.CalledProcessError:
        print("Error: Not in a git repository or git command failed")
        sys.exit(1)


def get_git_root(path=".") -> Path | None:
    """Return the repository root by walking up until a .git directory is found."""
    path = os.path.abspath(path)

    while True:
        if os.path.isdir(os.path.join(path, ".git")):
            return Path(path)
        newpath = os.path.dirname(path)
        if newpath == path:  # reached filesystem root
            return None
        path = newpath


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


def get_commit_message_file() -> str | None:
    """Read the content of .git/COMMIT_MESSAGE."""
    gitroot = get_git_root()
    if gitroot is None:
        print("No gitroot")
        return None

    commit_message_file = gitroot / ".git" / "COMMIT_EDITMSG"
    if commit_message_file.exists():
        return "\n".join(
            [
                l
                for l in commit_message_file.read_text().splitlines()
                if not l.startswith("#")
            ]
        ).strip()

    return None


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
            max_tokens=2000,
            extra_headers={
                "HTTP-Referer": "https://mgalgs.io",
                "X-Title": "Commitothy",
            },
        )

        return {
            "content": response.choices[0].message.content.strip(),
            "model": response.model,
        }
    except Exception as e:
        if debug:
            print(f"Error calling OpenRouter API: {e}")
        return None


def generate_commit_message(
    diff,
    recent_messages,
    model="openrouter/auto",
    debug=False,
    num_retries=3,
    improve_message=False,
    cursor_position=None,
):
    """Generate or improve a commit message using OpenAI."""
    examples_text = (
        "\n---\n".join(recent_messages) if recent_messages else "No examples available"
    )

    if improve_message:
        commit_message = get_commit_message_file()
        if not commit_message:
            print("Error: No existing commit message found in .git/COMMIT_MESSAGE")
            return None

        with_cursor_position_short = ""
        with_cursor_position = ""
        replace_cursor_position = ""
        where_to_add = "to the end"
        if cursor_position is not None:
            if cursor_position < 0 or cursor_position > len(commit_message):
                print("Error: Invalid cursor position")
                return None
            with_cursor_position_short = " with cursor position"
            with_cursor_position = (
                " with a cursor position marked by `<CURSOR_IS_HERE>`"
            )
            replace_cursor_position = "Replace `<CURSOR_IS_HERE>` with the new content."
            commit_message = (
                commit_message[:cursor_position]
                + "<CURSOR_IS_HERE>"
                + commit_message[cursor_position:]
            )
            where_to_add = "to the cursor position"

        extra_context = f"""You need to improve an existing commit message that the user has already started on. Preserve the user's content exactly, only add content to {where_to_add}. Existing commit message{with_cursor_position_short}:
<message>
{commit_message}
</message>
"""

        preamble = f"Below is an existing commit message that you should improve{with_cursor_position}."
        postamble = replace_cursor_position
        prompt = PROMPT_TEMPLATE.format(
            diff=diff,
            examples_text=examples_text,
            extra_context=extra_context,
            preamble=preamble,
            postamble=postamble,
        )
    else:
        preamble = ""
        postamble = ""
        prompt = PROMPT_TEMPLATE.format(
            diff=diff,
            examples_text=examples_text,
            extra_context="",
            preamble=preamble,
            postamble=postamble,
        )

    prompt = prompt.strip()

    if debug:
        print("Debug mode enabled. Prompt being sent to OpenRouter:")
        print(prompt)

    while num_retries > 0:
        response = llm_call(prompt, model=model, debug=debug)
        if response and response.get("content"):
            return {
                "message": response["content"].strip(),
                "model": response["model"],
            }
        num_retries -= 1

    return None


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate git commit messages using AI"
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen3-235b-a22b-2507",
        help="Model to use for generation (default: qwen/qwen3-235b-a22b-2507)",
    )
    parser.add_argument(
        "--head",
        action="store_true",
        help="Analyze changes from `git show HEAD` rather than `git diff --staged`. Useful when amending.",
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
    parser.add_argument(
        "--no-trailer",
        action="store_true",
        help="Don't add model name as a trailer in the commit message",
    )
    parser.add_argument(
        "--improve-message",
        action="store_true",
        help="Improve an existing commit message from .git/COMMIT_MESSAGE",
    )
    parser.add_argument(
        "--improve-message-cursor-position",
        type=int,
        help="Cursor position in .git/COMMIT_MESSAGE for completion (requires --improve-message)",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it with: export OPENROUTER_API_KEY=your_api_key_here")
        sys.exit(1)

    diff = git(["show", "HEAD"]) if args.head else git(["diff", "--staged"])

    if not diff.strip():
        print("No staged changes found. Please stage some changes first.")
        sys.exit(1)

    files = get_changed_files()

    recent_messages = get_recent_messages(files, limit=args.history_limit)

    result = generate_commit_message(
        diff,
        recent_messages,
        args.model,
        debug=args.debug,
        num_retries=args.num_retries,
        improve_message=args.improve_message,
        cursor_position=args.improve_message_cursor_position,
    )

    if not result:
        print("Failed to generate commit message after retries.")
        sys.exit(1)

    commit_message = result["message"]

    if not args.no_trailer:
        trailer = f"Commit-Message-Co-Author: {result['model']}"
        last_line = commit_message.split("\n")[-1].strip()
        if re.match(r"^[^\s]+:\s", last_line):
            # Message already has trailers, append ours to the list
            commit_message += f"\n{trailer}"
        else:
            # No trailers, add ours as a new line
            commit_message += f"\n\n{trailer}"

    print(commit_message)


if __name__ == "__main__":
    main()
