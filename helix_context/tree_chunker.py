"""
Tree-sitter AST chunking — boundary-aware code splitting.

Optional backend for CodonChunker._chunk_code. Uses tree-sitter to
split code along real AST boundaries (functions, classes, methods)
instead of regex-matched keywords.

Biology:
    The regex chunker cuts wherever it sees 'def' or 'class' — like
    a restriction enzyme that can't tell a real cut site from a
    sequence that happens to look like one. Tree-sitter understands
    the grammar: it cuts at function/class definitions, not at 'def'
    inside a docstring or a variable named 'my_class_name'.

Usage (from CodonChunker):
    try:
        from .tree_chunker import chunk_code_ast
        strands = chunk_code_ast(code, max_chars, language="python")
    except ImportError:
        strands = _chunk_code_regex(code, max_chars)  # Fallback

Supported languages (tier 1):
    - python
    - rust
    - javascript
    - typescript

Install:
    pip install tree-sitter tree-sitter-python tree-sitter-rust \
                tree-sitter-javascript tree-sitter-typescript
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("helix.tree_chunker")


# ── Language detection from file path ────────────────────────────

_EXT_TO_LANG = {
    ".py": "python",
    ".pyw": "python",
    ".rs": "rust",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp",
    ".rb": "ruby",
    ".lua": "lua",
}


def detect_language(source_id: Optional[str]) -> Optional[str]:
    """Infer tree-sitter language from source file extension."""
    if not source_id:
        return None
    import os
    _, ext = os.path.splitext(source_id.lower())
    return _EXT_TO_LANG.get(ext)


# ── Parser cache ─────────────────────────────────────────────────

_parser_cache: Dict[str, "object"] = {}


def _get_parser(language: str):
    """
    Load and cache a tree-sitter parser for the given language.

    Uses individual tree-sitter-<lang> packages (tree-sitter-python, etc.)
    instead of the deprecated tree-sitter-languages bundle, which isn't
    maintained for Python 3.12+.

    Raises ImportError if tree-sitter stack isn't installed.
    """
    if language in _parser_cache:
        return _parser_cache[language]

    try:
        from tree_sitter import Language, Parser
    except ImportError as e:
        raise ImportError(
            "tree-sitter chunking requires: pip install tree-sitter "
            "tree-sitter-python tree-sitter-rust tree-sitter-javascript "
            "tree-sitter-typescript"
        ) from e

    # Map language name to module import
    lang_module_map = {
        "python": ("tree_sitter_python", "language"),
        "rust": ("tree_sitter_rust", "language"),
        "javascript": ("tree_sitter_javascript", "language"),
        "typescript": ("tree_sitter_typescript", "language_typescript"),
        "tsx": ("tree_sitter_typescript", "language_tsx"),
    }

    if language not in lang_module_map:
        raise ImportError(f"No tree-sitter package configured for {language}")

    module_name, attr_name = lang_module_map[language]
    try:
        import importlib
        mod = importlib.import_module(module_name)
        lang_capsule = getattr(mod, attr_name)()
    except ImportError as e:
        raise ImportError(
            f"Missing tree-sitter-{language}: pip install {module_name.replace('_', '-')}"
        ) from e

    lang = Language(lang_capsule)
    parser = Parser(lang)
    _parser_cache[language] = parser
    return parser


# ── Boundary node types per language ─────────────────────────────
#
# These are the AST node types that mark "good cut points" — places
# where splitting preserves semantic coherence. Functions and classes
# are universal; the rest are language-specific.

_BOUNDARY_NODES: Dict[str, Tuple[str, ...]] = {
    "python": (
        "function_definition",
        "class_definition",
        "decorated_definition",
    ),
    "rust": (
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
        "mod_item",
    ),
    "javascript": (
        "function_declaration",
        "class_declaration",
        "method_definition",
        "arrow_function",
        "export_statement",
    ),
    "typescript": (
        "function_declaration",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "export_statement",
    ),
    "tsx": (
        "function_declaration",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "export_statement",
    ),
    "go": (
        "function_declaration",
        "method_declaration",
        "type_declaration",
    ),
    "java": (
        "method_declaration",
        "class_declaration",
        "interface_declaration",
    ),
    "c": (
        "function_definition",
        "struct_specifier",
    ),
    "cpp": (
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "namespace_definition",
    ),
    "ruby": (
        "method",
        "class",
        "module",
    ),
    "lua": (
        "function_declaration",
        "local_function",
    ),
}


# ── AST-aware chunking ───────────────────────────────────────────

def chunk_code_ast(
    code: str,
    max_chars: int = 4000,
    language: Optional[str] = None,
    source_id: Optional[str] = None,
) -> List[Tuple[str, bool]]:
    """
    Split code along AST boundaries, respecting max_chars size limit.

    Args:
        code: Source code as string
        max_chars: Target max size per chunk (soft limit — respects AST)
        language: Tree-sitter language name (e.g., "python")
        source_id: File path for language auto-detection (if language=None)

    Returns:
        List of (chunk_content, is_fragment) tuples. is_fragment=True
        means the chunk had to be hard-cut because a single AST node
        exceeded max_chars (e.g., a huge function).

    Raises:
        ImportError: if tree-sitter isn't installed
        ValueError: if language is unknown/unsupported
    """
    # Language resolution
    if language is None:
        language = detect_language(source_id)
    if language is None or language not in _BOUNDARY_NODES:
        # Unknown language — cannot AST-chunk safely, caller should fall back
        raise ValueError(
            f"Unsupported or undetected language for tree-sitter chunking "
            f"(source={source_id!r}, language={language!r})"
        )

    parser = _get_parser(language)
    tree = parser.parse(code.encode("utf-8"))
    root = tree.root_node

    boundary_types = set(_BOUNDARY_NODES[language])

    # Collect top-level boundary nodes (functions, classes, etc.)
    # Along with interstitial content (imports, module-level statements).
    blocks: List[Tuple[str, bool]] = []
    code_bytes = code.encode("utf-8")

    def node_text(node) -> str:
        """Extract UTF-8 text for an AST node."""
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    # Walk direct children of the root, collecting boundary blocks
    # in source order. Non-boundary siblings (imports, module-level code)
    # get grouped into a "prelude" or "interstitial" block.
    prelude_end = 0
    for child in root.children:
        if child.type in boundary_types:
            # Flush any prelude content before this boundary
            if child.start_byte > prelude_end:
                prelude_text = code_bytes[prelude_end:child.start_byte].decode(
                    "utf-8", errors="replace"
                )
                if prelude_text.strip():
                    blocks.append((prelude_text, False))
            # Add the boundary block itself
            block_text = node_text(child)
            blocks.append((block_text, False))
            prelude_end = child.end_byte
        # Non-boundary children accumulate into the next prelude flush

    # Trailing content after the last boundary
    if prelude_end < len(code_bytes):
        tail = code_bytes[prelude_end:].decode("utf-8", errors="replace")
        if tail.strip():
            blocks.append((tail, False))

    # If no boundaries found at all, return the whole file as one block
    if not blocks:
        blocks = [(code, False)]

    # Merge small adjacent blocks to hit max_chars, hard-cut huge ones
    merged: List[Tuple[str, bool]] = []
    current = ""

    for block, _ in blocks:
        if len(current) + len(block) < max_chars:
            current += block
        else:
            if current:
                merged.append((current.strip(), False))
                current = ""

            if len(block) >= max_chars:
                # Block itself is too big — hard cut at max_chars
                remaining = block
                while len(remaining) >= max_chars:
                    merged.append((remaining[:max_chars], True))
                    remaining = remaining[max_chars:]
                current = remaining
            else:
                current = block

    if current.strip():
        merged.append((current.strip(), False))

    return merged


def is_available() -> bool:
    """Check whether tree-sitter chunking is usable in this environment."""
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_python  # noqa: F401
        return True
    except ImportError:
        return False
