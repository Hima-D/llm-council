# upload_to_huggingface.py
# Cleaned script to upload your local 'llm-council' repo to Hugging Face as a Space (for demo) or generic repo.
# Assumes:
# - You're in the 'llm-council' directory.
# - HF_TOKEN set as env var (get from https://huggingface.co/settings/tokens; write access required).
# - huggingface_hub installed: pip install huggingface_hub
# - For Space: Adds a basic app.py for Gradio demo (wraps council.deliberate).
# Run: python upload_to_huggingface.py
#
# Changes for cleanliness:
# - Updated for huggingface_hub v1.0+ (replaced deprecated HfFolder with hf_login).
# - Removed unused imports (shutil, Repository).
# - Improved file exclusion logic.
# - Fixed JSON in app.py (single braces).
# - Better error handling and logging.
# - Simplified repo name handling (no global mutation).

import os
from pathlib import Path
from typing import List

from huggingface_hub import (
    HfApi,
    create_repo,
    upload_file,
    login,  # Modern auth method
)

# Config
REPO_NAME = "llm-council"  # Base name; script appends suffix if needed.
REPO_TYPE = "space"  # Options: "space" (Gradio demo), "model" (generic repo), "dataset".
SPACE_SDK = "gradio"  # SDK for Space.
EXCLUDED_DIRS = {"venv", "audit_logs", "__pycache__", ".git"}  # Directories to skip.
EXCLUDED_PATTERNS = {".bak", ".pyc"}  # File patterns to skip (ends with).


def create_gitignore(excludes: List[str]) -> None:
    """Create .gitignore if missing."""
    gitignore_path = Path(".gitignore")
    if not gitignore_path.exists():
        with open(gitignore_path, "w") as f:
            f.write("\n".join(excludes) + "\n")
        print("Created .gitignore")


def generate_app_py(council_module: str = "council") -> None:
    """Generate basic Gradio app.py for Space demo."""
    app_content = f'''import gradio as gr
from {council_module} import AdvancedLLMCouncil  # Adjust import as needed
import os
import json

def deliberate_query(question):
    api_key = os.getenv("OPENAI_API_KEY")  # User sets in Space secrets
    if not api_key:
        return {{"error": "Set OPENAI_API_KEY in Space secrets."}}
    
    council = AdvancedLLMCouncil(api_key)
    decision = council.deliberate(question, max_reflection_rounds=1)
    return json.dumps(decision.model_dump(), indent=2)

iface = gr.Interface(
    fn=deliberate_query, 
    inputs=gr.Textbox(label="Query", placeholder="e.g., What is ML?"),
    outputs=gr.JSON(label="Decision Object"),
    title="LLM Council Demo",
    description="3 agents + 2 judges deliberate your query."
)

if __name__ == "__main__":
    iface.launch()
'''
    with open("app.py", "w") as f:
        f.write(app_content)
    print("Generated app.py for Gradio Space")


def get_files_to_upload() -> List[str]:
    """Walk directory and collect files, excluding patterns."""
    files = []
    for root, dirs, filenames in os.walk("."):
        # Skip excluded dirs in-place
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        
        for filename in filenames:
            if not any(filename.endswith(pat) for pat in EXCLUDED_PATTERNS):
                rel_path = Path(root) / filename
                files.append(str(rel_path.relative_to(".")))
    return files


def main() -> None:
    """Main upload workflow."""
    # Authenticate
    token = os.getenv("HF_TOKEN")
    if not token:
        print("Error: Set HF_TOKEN env var (from https://huggingface.co/settings/tokens).")
        return
    try:
        login(token=token)
        api = HfApi()
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Logged in as: {username}")
    except Exception as e:
        print(f"Auth error: {e}")
        return

    # Determine repo name (avoid conflicts)
    repo_name = REPO_NAME
    try:
        api.repo_info(f"{username}/{repo_name}", repo_type=REPO_TYPE)
        repo_name = f"{REPO_NAME}-upload"  # Suffix for new
        print(f"Repo '{REPO_NAME}' exists. Using '{repo_name}' instead.")
    except Exception:
        pass  # Fresh repo OK

    full_repo_id = f"{username}/{repo_name}"

    # Create repo
    try:
        create_repo(
            full_repo_id,
            repo_type=REPO_TYPE,
            space_sdk=SPACE_SDK if REPO_TYPE == "space" else None,
            exist_ok=True,
        )
        print(f"Created/Verified {REPO_TYPE}: https://huggingface.co/{full_repo_id}")
    except Exception as e:
        print(f"Repo creation error: {e}")
        return

    # Prepare local files
    create_gitignore(list(EXCLUDED_DIRS | EXCLUDED_PATTERNS))
    if REPO_TYPE == "space":
        generate_app_py()

    local_files = get_files_to_upload()

    # Upload files
    for local_file in local_files:
        try:
            upload_file(
                path_or_fileobj=local_file,
                path_in_repo=local_file,
                repo_id=full_repo_id,
                repo_type=REPO_TYPE,
                commit_message=f"Upload {Path(local_file).name}",
            )
            print(f"Uploaded: {local_file}")
        except Exception as e:
            print(f"Upload failed for {local_file}: {e}")

    # Final notes
    print(f"\n✅ Upload complete! View at: https://huggingface.co/{full_repo_id}")
    if REPO_TYPE == "space":
        print("💡 For demo: Go to Space > Settings > Secrets > Add OPENAI_API_KEY.")
    print("🗑️ Clean up: rm app.py .gitignore (if auto-generated)")


if __name__ == "__main__":
    main()