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


def get_recent_commits(files, limit=20):
    """Get recent commit messages for the specified files."""
    if not files:
        return []

    try:
        cmd = ["git", "log", f"-{limit}", "--pretty=format:%B"] + files
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        commits = result.stdout.strip().split("\n\n")
        return [commit for commit in commits if commit.strip()]
    except subprocess.CalledProcessError:
        return []


def analyze_commit_patterns(commits):
    """Analyze recent commits to understand the commit message pattern."""
    summaries = []
    bodies = []

    for commit in commits:
        lines = commit.strip().split("\n")
        if lines:
            summaries.append(lines[0])
            if len(lines) > 1:
                bodies.append("\n".join(lines[1:]).strip())

    return summaries, bodies


def generate_commit_message(diff, summaries, bodies, model="openrouter/auto"):
    """Generate a commit message using OpenAI."""
    # Create examples from recent commits
    examples = []
    for i, summary in enumerate(summaries):
        body = bodies[i] if i < len(bodies) else ""
        example = summary
        if body:
            example += f"\n{body}"
        examples.append(example)

    examples_text = (
        "\n\n---\n\n".join(examples) if examples else "No examples available"
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

Examples of recent commits:
{examples_text}

Git diff to analyze:
{diff}

Generate only the commit message with no other text."""

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
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenRouter API: {e}")
        sys.exit(1)


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
    args = parser.parse_args()

    # Check if we have an API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Please set it with: export OPENROUTER_API_KEY=your_api_key_here")
        sys.exit(1)

    # Get staged changes
    diff = get_staged_diff()

    if not diff.strip():
        print("No staged changes found. Please stage some changes first.")
        sys.exit(1)

    # Get changed files
    files = get_changed_files()

    # Get recent commit messages for these files
    recent_commits = get_recent_commits(files, limit=args.history_limit)

    # Analyze commit patterns
    summaries, bodies = analyze_commit_patterns(recent_commits)

    # Generate commit message
    commit_message = generate_commit_message(diff, summaries, bodies, args.model)

    # Output the commit message
    print(commit_message)


if __name__ == "__main__":
    main()
