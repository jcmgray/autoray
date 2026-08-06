#!/usr/bin/env python3
"""Resolve changelog references to full markdown links.

This script converts MyST/Sphinx-style roles in the changelog, such as
``{func}`~package.function``` or ``{issue}`20``` to normal markdown links
that can be pasted directly into GitHub release notes.

Usage:
    python resolve_changelog.py input.md [output.md]

If ``output.md`` is not given, prints to stdout.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import re
import sys
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = DOCS_DIR.parent


def get_project_metadata():
    """Read the package name and public URLs from ``pyproject.toml``."""
    text = (PROJECT_DIR / "pyproject.toml").read_text(encoding="utf-8")

    def _find(pattern):
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match is None:
            raise ValueError("Missing required project metadata.")
        return match.group(1).rstrip("/")

    package = _find(r'^name\s*=\s*"([^"]+)"')
    docs_url = _find(r'^"Documentation"\s*=\s*"([^"]+)"')
    repository_url = _find(r'^"Repository"\s*=\s*"([^"]+)"')
    return package, f"{docs_url}/en/latest", repository_url


PACKAGE, BASE, REPO_BASE = get_project_metadata()
API_BASE = f"{BASE}/autoapi"

OBJECT_ROLES = {
    "attr",
    "class",
    "const",
    "data",
    "exc",
    "func",
    "meth",
    "mod",
    "obj",
}
CALLABLE_ROLES = {"func", "meth"}
DOC_SUFFIXES = {".ipynb", ".md", ".rst"}
RESOLUTION_ERRORS = (AttributeError, ImportError, ValueError)


def is_package_reference(target):
    """Check whether a target names the current package or one of its objects."""
    return target == PACKAGE or target.startswith(f"{PACKAGE}.")


def split_explicit_title(value):
    """Split `title <target>` syntax used by Sphinx/MyST roles."""
    match = re.fullmatch(r"(.+?)\s*<([^<>]+)>", value)
    if match is None:
        return None, value.strip()
    return match.group(1).strip(), match.group(2).strip()


def display_name(role, raw_target):
    """Infer how a role would typically render in the docs."""
    explicit_title, target = split_explicit_title(raw_target)
    if explicit_title is not None:
        return explicit_title, target

    target = target.strip()
    shortened = target.startswith("~")
    canonical = target.lstrip("~")

    if role == "doc":
        label = canonical.split("#", 1)[0].rstrip("/")
        return label.rsplit("/", 1)[-1], canonical

    if role == "issue":
        return f"#{canonical}", canonical

    if role in {"pr", "pull"}:
        return f"#{canonical}", canonical

    if shortened:
        label = canonical.rsplit(".", 1)[-1]
    else:
        label = canonical

    if role in CALLABLE_ROLES and not label.endswith("()"):
        label = f"{label}()"

    return label, canonical


@functools.cache
def import_dotted_name(name):
    """Import a dotted Python name by finding the longest module prefix."""
    parts = name.split(".")

    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            obj = importlib.import_module(module_name)
        except ImportError:
            continue

        for attr in parts[index:]:
            obj = getattr(obj, attr)
        return obj

    raise ImportError(f"Could not import '{name}'.")


@functools.cache
def object_to_url(name):
    """Convert a dotted package object reference to an AutoAPI URL."""
    obj = import_dotted_name(name)

    if inspect.ismodule(obj):
        module_name = obj.__name__
        return f"{API_BASE}/{module_name.replace('.', '/')}/index.html"

    module = inspect.getmodule(obj)
    qualname = getattr(obj, "__qualname__", None)

    if module is None or qualname is None:
        raise ValueError(f"Could not resolve documentation target '{name}'.")

    module_name = module.__name__
    anchor = f"{module_name}.{qualname}"
    module_path = module_name.replace(".", "/")
    return f"{API_BASE}/{module_path}/index.html#{anchor}"


@functools.lru_cache(maxsize=1)
def discover_doc_pages():
    """Collect known documentation source pages by docname."""
    doc_pages = {}

    for path in DOCS_DIR.rglob("*"):
        if not path.is_file() or path.suffix not in DOC_SUFFIXES:
            continue

        relative = path.relative_to(DOCS_DIR)
        if any(part.startswith("_") for part in relative.parts):
            continue
        if relative.parts[0] == "utils":
            continue

        docname = relative.with_suffix("").as_posix()
        doc_pages[docname] = docname

        basename = relative.stem
        existing = doc_pages.get(basename)
        if existing is None or existing == basename:
            doc_pages[basename] = docname

    return doc_pages


@functools.cache
def doc_to_url(target):
    """Convert a doc role target to an absolute documentation URL."""
    page, _, anchor = target.partition("#")
    page = page.rstrip("/") or "index"
    if Path(page).suffix in DOC_SUFFIXES:
        page = Path(page).with_suffix("").as_posix()
    doc_pages = discover_doc_pages()
    docname = doc_pages.get(page, page)
    suffix = f"#{anchor}" if anchor else ""
    return f"{BASE}/{docname}.html{suffix}"


def role_to_url(role, target):
    """Resolve a single MyST/Sphinx role target to a URL."""
    if role in OBJECT_ROLES:
        if not is_package_reference(target):
            raise ValueError(f"Unsupported object target '{target}'.")
        return object_to_url(target)

    if role == "doc":
        return doc_to_url(target)

    if role == "issue":
        return f"{REPO_BASE}/issues/{target}"

    if role in {"pr", "pull"}:
        return f"{REPO_BASE}/pull/{target}"

    raise ValueError(f"Unsupported role '{role}'.")


def resolve_markdown_links(text):
    """Resolve local documentation and package object markdown links."""

    def _replace(match):
        link_text = match.group(1)
        target = match.group(2).strip()

        try:
            if is_package_reference(target):
                url = object_to_url(target)
            else:
                page = target.partition("#")[0]
                if Path(page).suffix not in DOC_SUFFIXES:
                    return match.group(0)
                docname = Path(page).with_suffix("").as_posix()
                if docname not in discover_doc_pages().values():
                    return match.group(0)
                url = doc_to_url(target)
        except RESOLUTION_ERRORS:
            return match.group(0)

        return f"[{link_text}]({url})"

    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", _replace, text)


def resolve_roles(text):
    """Resolve MyST/Sphinx role syntax to markdown links."""

    def _replace(match):
        role = match.group(1)
        raw_target = match.group(2)

        try:
            label, target = display_name(role, raw_target)
            url = role_to_url(role, target)
        except RESOLUTION_ERRORS:
            return match.group(0)

        return f"[{label}]({url})"

    return re.sub(r"\{([a-zA-Z:]+)\}`([^`]+)`", _replace, text)


def resolve_links(text):
    """Resolve supported references in markdown text."""
    text = resolve_markdown_links(text)
    text = resolve_roles(text)
    return text


def main():
    if len(sys.argv) < 2:
        print(__doc__.strip())
        sys.exit(1)

    with open(sys.argv[1], encoding="utf-8") as file_obj:
        text = file_obj.read()

    result = resolve_links(text)

    if len(sys.argv) >= 3:
        with open(sys.argv[2], "w", encoding="utf-8") as file_obj:
            file_obj.write(result)
    else:
        print(result)


if __name__ == "__main__":
    main()
