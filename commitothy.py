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


PROMPT_TEMPLATE = """\
You are a commit message generator. Based on the context below (git diff, examples of recent commit messages for these files, etc), generate an appropriate commit message.

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
{recent_patches_text}

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

REVIEW_PROMPT_TEMPLATE = """\
You are a senior code reviewer. Produce a TL;DR-first review that a busy developer can scan in seconds, while still offering details for deeper reading when needed.

Mode: {mode}

Critical priorities:
- If there are security issues, make a huge fuss in the TL;DR and detail them first.
- If there are clear bugs, call them out explicitly in the TL;DR and detail them.
- If nothing critical, keep the TL;DR reassuring and concise.

Output format (strict):
TL;DR: [TAG,...] <1-2 sentence summary>
# TAG must be one of: [SECURITY], [BUG], [ISSUE], [OK].
Text should wrap at 80 characters.

Details:
- Security: <bullets, or "None">
- Bugs: <bullets, or "None">
- Correctness & Edge Cases: <bullets, or "None">
- Performance: <bullets, or "None">
- Tests: <bullets, or "Suggested test updates">
- Maintainability/Style: <bullets, or "None">

Guidelines:
{mode_instructions}
- Be specific, reference lines/hunks when possible.
- Professional, helpful tone. No preface or closing beyond the required sections.
- Include code when recommendations are provided. Including code is extremely
  helpful for busy developers looking to incorporate review feedback quickly.

Context to review:

Git diff:
<diff>
{diff}
</diff>

{file_context}

{recent_patches}
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


def get_touched_files(head: bool = False):
    """Get list of files touched in the current context (staged or HEAD)."""
    if head:
        try:
            result = subprocess.run(
                ["git", "show", "--name-only", "--pretty=format:", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            files = [f for f in result.stdout.strip().split("\n") if f.strip()]
            return files
        except subprocess.CalledProcessError:
            return []
    return get_changed_files()


def _get_recent_commits(
    files, limit, git_log_args: list[str] | None = None
) -> list[str]:
    """Get recent commit messages for the specified files."""
    if not files:
        return []

    git_log_args = git_log_args or []
    try:
        separator = "===COMMIT_SEPARATOR==="
        cmd = (
            [
                "git",
                "log",
                f"-{limit}",
                f"--pretty=format:{separator}%B",
            ]
            + git_log_args
            + ["--"]
            + files
        )
        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", check=True)
        commits = result.stdout.strip().split(separator)
        return [commit for commit in commits if commit.strip()]
    except subprocess.CalledProcessError:
        return []


def get_recent_messages(files, limit=20):
    return _get_recent_commits(files, limit)


def get_recent_patches(files: list[str], limit=5) -> list[str]:
    """
    Get the 5 most recent patches that touched the specified files. Anything
    longer than 500 lines is elided in the middle.
    """
    if not files:
        return []

    patches = _get_recent_commits(files, limit, git_log_args=["--patch"])
    ret_patches = []
    for patch in patches:
        # Elide to 500 lines (center elision)
        lines = patch.splitlines()
        if len(lines) > 500:
            lines = (
                lines[:250]
                + [
                    "",
                    f"### PATCH ELIDED FOR BREVITY ({len(lines) - 500} lines skipped) ###",
                    "",
                ]
                + lines[-250:]
            )
        ret_patches.append("\n".join(lines))

    return ret_patches


def format_recent_patches(patches: list[str]) -> str:
    return (
        "<recent_patches>"
        + "".join(f"\n<patch>\n{patch}\n</patch>\n" for patch in patches)
        + "</recent_patches>"
    )


def get_commit_message_file() -> str | None:
    """Read the content of .git/COMMIT_MESSAGE."""
    gitroot = get_git_root()
    if gitroot is None:
        print("No gitroot")
        return None

    commit_message_file = gitroot / ".git" / "COMMIT_EDITMSG"
    if commit_message_file.exists():
        return commit_message_file.read_text()

    return None


def llm_call(prompt, model="openrouter/auto", debug=False, max_tokens=2000):
    """Call the OpenRouter API to generate text."""
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
                    "content": "You are a helpful assistant that writes git commit messages and code reviews.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=max_tokens,
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
    recent_messages: list[str],
    recent_patches: list[str],
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

    recent_patches_text = ""
    if recent_patches:
        recent_patches_text = (
            "Recent patches touching these files:\n"
            + format_recent_patches(recent_patches)
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
                print(
                    f"Error: Invalid cursor position ({cursor_position}, len={len(commit_message)})"
                )
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
            recent_patches_text=recent_patches_text,
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
            recent_patches_text=recent_patches_text,
            extra_context="",
            preamble=preamble,
            postamble=postamble,
        )

    prompt = prompt.strip()

    if debug:
        print("Debug mode enabled. Prompt being sent to OpenRouter:")
        print(prompt)

    retries = num_retries
    while retries > 0:
        response = llm_call(prompt, model=model, debug=debug, max_tokens=2000)
        if response and response.get("content"):
            return {
                "message": response["content"].strip(),
                "model": response["model"],
            }
        retries -= 1

    return None


def collect_full_file_context(files: list[str]) -> str:
    gitroot = get_git_root()
    if gitroot is None or not files:
        return ""
    chunks: list[str] = []
    for f in files:
        try:
            p = (gitroot / f).resolve()
            if not p.is_file():
                continue
            content = p.read_text(encoding="utf-8", errors="replace")
            chunks.append(
                f'<file path="{f}">\n<content>\n{content}\n</content>\n</file>'
            )
        except Exception:
            continue
    if not chunks:
        return ""
    return (
        "Full contents of touched files (for additional context):\n<files>\n"
        + "\n\n".join(chunks)
        + "\n</files>\n"
    )


def generate_code_review(
    diff: str,
    files: list[str],
    mode: str,
    model: str,
    debug: bool,
    include_full_file_context: bool,
    recent_patches_to_consider: list[str],
    num_retries: int = 3,
):
    file_context = ""
    if include_full_file_context:
        file_context = collect_full_file_context(files)

    recent_patches = ""
    if recent_patches_to_consider:
        recent_patches = (
            "Recent patches that touched the files touched in this patch:\n"
            + format_recent_patches(recent_patches_to_consider)
        )

    if mode == "quick":
        mode_instructions = (
            "- Keep the body to <= 8 short bullets total across sections.\n"
            "- Skip nitpicks unless they materially reduce clarity or safety.\n"
            "- Prefer actionable phrasing (e.g., 'Add X test', 'Handle Y edge case')."
        )
    else:
        mode_instructions = (
            "- Provide thorough, but organized sections with concise bullets.\n"
            "- Include concrete suggestions and brief rationale; cite hunks/lines when possible.\n"
            "- It's acceptable for this to be longer, but avoid repetition."
        )

    prompt = REVIEW_PROMPT_TEMPLATE.format(
        mode=mode,
        mode_instructions=mode_instructions,
        diff=diff,
        file_context=file_context,
        recent_patches=recent_patches,
    ).strip()

    if debug:
        print("Debug mode enabled. Code review prompt being sent to OpenRouter:")
        print(prompt)

    retries = num_retries
    max_tokens = 3500 if mode == "full" else 1200
    while retries > 0:
        response = llm_call(prompt, model=model, debug=debug, max_tokens=max_tokens)
        if response and response.get("content"):
            return response["content"].strip()
        retries -= 1
    return None


def comment_lines(text: str) -> str:
    return "\n".join(["# " + line for line in text.splitlines()])


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
    parser.add_argument(
        "--code-review",
        nargs="?",
        const="full",
        choices=["quick", "full"],
        help=(
            "Append an AI code review as commented markdown after the commit message. "
            "If provided without a value, defaults to 'quick'. Use 'full' to include full contents of touched files for deeper context."
        ),
    )
    parser.add_argument(
        "--consider-recent-patches",
        action="store_true",
        help="Include recent patches in LLM context during commit message generation and code review",
    )
    parser.add_argument(
        "--include-full-file-context",
        action="store_true",
        help="Include full file contents in LLM context during commit message generation and code review",
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

    files = get_touched_files(args.head)

    recent_messages = get_recent_messages(files, limit=args.history_limit)
    recent_patches = get_recent_patches(files) if args.consider_recent_patches else []

    result = generate_commit_message(
        diff,
        recent_messages,
        recent_patches,
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

    if args.code_review:
        mode = args.code_review
        review_text = generate_code_review(
            diff=diff,
            files=files,
            mode=mode,
            model=args.model,
            debug=args.debug,
            include_full_file_context=args.include_full_file_context,
            recent_patches_to_consider=recent_patches,
            num_retries=args.num_retries,
        )
        if review_text:
            header = f"*** CODE REVIEW ({mode}) ***"
            hr = "=" * len(header)
            print(f"\n# {hr}\n# {header}\n# {hr}\n")
            print(comment_lines(review_text))


if __name__ == "__main__":
    main()
