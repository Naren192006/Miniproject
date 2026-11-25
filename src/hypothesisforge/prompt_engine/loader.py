"""
HypothesisForge - Prompt Loader
Loads versioned prompt Markdown files, extracts metadata and system prompts,
and renders user prompt templates with strict placeholder validation.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


@dataclass
class PromptDefinition:
    name: str
    file_path: str
    version: str
    stage: str
    purpose: str
    system_prompt: str
    user_template: str
    placeholders: Set[str] = field(default_factory=set)

    def render_user_prompt(self, **kwargs: Any) -> str:
        """
        Renders the user prompt template by substituting {{placeholder}} values.
        Validates that all required placeholders are provided.
        """
        missing = [p for p in self.placeholders if p not in kwargs]
        if missing:
            raise ValueError(
                f"Missing required placeholder values for prompt '{self.name}': {', '.join(missing)}"
            )

        rendered = self.user_template
        for key, value in kwargs.items():
            pattern = re.compile(r"\{\{\s*" + re.escape(key) + r"\s*\}\}")
            rendered = pattern.sub(str(value), rendered)

        # Ensure no unrendered {{...}} placeholders remain
        remaining = re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", rendered)
        if remaining:
            raise ValueError(
                f"Unrendered placeholders remaining in prompt '{self.name}': {', '.join(set(remaining))}"
            )

        return rendered


class PromptLoader:
    """
    Manages loading and version tracking for all prompts in the prompts/ directory.
    """

    def __init__(self, prompts_dir: Optional[Path] = None):
        if prompts_dir is None:
            # Default to <workspace>/prompts
            base_dir = Path(__file__).resolve().parent.parent.parent.parent
            prompts_dir = base_dir / "prompts"
        self.prompts_dir = Path(prompts_dir)

    def get_prompt_path(self, identifier: str) -> Path:
        """
        Resolves identifier (e.g. '01', '01_literature_extraction', '01_literature_extraction.md')
        to full Path in prompts directory.
        """
        # If full file path or filename already provided
        if identifier.endswith(".md"):
            direct_path = self.prompts_dir / identifier
            if direct_path.exists():
                return direct_path

        # If starts with digits e.g. "01"
        for p in self.prompts_dir.glob("*.md"):
            stem = p.stem.lower()
            ident_clean = identifier.lower().replace(".md", "")
            if stem == ident_clean or stem.startswith(f"{ident_clean}_"):
                return p

        raise FileNotFoundError(f"Prompt file matching '{identifier}' not found in {self.prompts_dir}")

    def load_prompt(self, identifier: str) -> PromptDefinition:
        """
        Loads and parses a prompt markdown file.
        """
        path = self.get_prompt_path(identifier)
        content = path.read_text(encoding="utf-8")

        # Parse frontmatter if present
        metadata: Dict[str, str] = {}
        body = content
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                frontmatter_text = parts[1]
                body = parts[2]
                for line in frontmatter_text.strip().splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        metadata[k.strip()] = v.strip().strip('"').strip("'")

        # Parse headings
        # Expecting:
        # ## System prompt
        # ...
        # ## User prompt (template)
        # ...
        system_prompt = ""
        user_template = ""

        sys_match = re.search(
            r"##\s*System prompt\s*\n(.*?)(?=\n##|\Z)",
            body,
            re.DOTALL | re.IGNORECASE
        )
        if sys_match:
            system_prompt = sys_match.group(1).strip()

        user_match = re.search(
            r"##\s*User prompt\s*(?:\(template\))?\s*\n(.*?)(?=\n##|\Z)",
            body,
            re.DOTALL | re.IGNORECASE
        )
        if user_match:
            user_template = user_match.group(1).strip()
            # If wrapped in code blocks, strip them
            if user_template.startswith("```") and user_template.endswith("```"):
                lines = user_template.splitlines()
                user_template = "\n".join(lines[1:-1]).strip()

        # If metadata not in frontmatter, try extracting from text header
        if not metadata.get("version"):
            v_match = re.search(r"Version\s+([\d\.]+)", body, re.IGNORECASE)
            if v_match:
                metadata["version"] = v_match.group(1)

        if not metadata.get("stage"):
            stage_match = re.search(r'Used by:\s*(?:stage\s*)?["\']?([^"\'\n\(]+)["\']?', body, re.IGNORECASE)
            if stage_match:
                metadata["stage"] = stage_match.group(1).strip()

        if not metadata.get("purpose"):
            purpose_match = re.search(r"Purpose:\s*([^\n]+)", body, re.IGNORECASE)
            if purpose_match:
                metadata["purpose"] = purpose_match.group(1).strip()

        name = metadata.get("name", path.stem)
        version = metadata.get("version", "1.0")
        stage = metadata.get("stage", "unknown")
        purpose = metadata.get("purpose", "")

        # Extract {{placeholders}}
        placeholders = set(re.findall(r"\{\{([a-zA-Z0-9_]+)\}\}", user_template))

        return PromptDefinition(
            name=name,
            file_path=str(path),
            version=version,
            stage=stage,
            purpose=purpose,
            system_prompt=system_prompt,
            user_template=user_template,
            placeholders=placeholders,
        )
