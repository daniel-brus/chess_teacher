"""Integration checks for Streamlit pages and Docker packaging.

These tests intentionally avoid executing page bodies (auth / UI side effects).
They verify first-party imports resolve and the image copies the runtime packages
pages need — the failure mode behind ``ModuleNotFoundError: streamlit_components``.
"""

from __future__ import annotations

import ast
import importlib
import re
from collections.abc import Iterable
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

_FIRST_PARTY_ROOTS = frozenset({
    "chess_teacher",
    "streamlit_pages",
    "streamlit_utils",
    "streamlit_components",
})

# Installed into the image via ``pip install .`` (src/), not a top-level COPY.
_PIP_INSTALLED_ROOTS = frozenset({"chess_teacher"})


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _streamlit_source_files(root: Path) -> list[Path]:
    files = [root / "streamlit_app.py"]
    files.extend(sorted((root / "streamlit_pages").glob("*.py")))
    files.extend(sorted((root / "streamlit_utils").rglob("*.py")))
    files.extend(sorted((root / "streamlit_components").rglob("*.py")))
    return [path for path in files if path.is_file()]


def _iter_imported_modules(path: Path) -> Iterable[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


def _is_first_party(module: str) -> bool:
    return module.split(".", 1)[0] in _FIRST_PARTY_ROOTS


def _first_party_imports(root: Path) -> set[str]:
    modules: set[str] = set()
    for path in _streamlit_source_files(root):
        for module in _iter_imported_modules(path):
            if _is_first_party(module):
                modules.add(module)
    return modules


def _copy_runtime_package_roots(modules: Iterable[str]) -> set[str]:
    roots: set[str] = set()
    for module in modules:
        root = module.split(".", 1)[0]
        if root not in _PIP_INSTALLED_ROOTS:
            roots.add(root)
    return roots


def _st_page_paths(app_path: Path) -> list[str]:
    tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
    paths: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        is_st_page = (
            isinstance(func, ast.Attribute)
            and func.attr == "Page"
            and isinstance(func.value, ast.Name)
            and func.value.id == "st"
        )
        if not is_st_page or not node.args:
            continue
        arg0 = node.args[0]
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            paths.append(arg0.value)
    return paths


@pytest.fixture(scope="module")
def project_root() -> Path:
    return _project_root()


@pytest.fixture(scope="module")
def first_party_modules(project_root: Path) -> set[str]:
    return _first_party_imports(project_root)


def test_streamlit_app_registers_existing_page_files(project_root: Path) -> None:
    app_path = project_root / "streamlit_app.py"
    page_paths = _st_page_paths(app_path)
    assert page_paths, "streamlit_app.py should register at least one st.Page(...)"
    missing = [path for path in page_paths if not (project_root / path).is_file()]
    assert not missing, f"st.Page paths missing on disk: {missing}"


def test_dockerfile_copies_streamlit_runtime_packages(
    project_root: Path,
    first_party_modules: set[str],
) -> None:
    dockerfile = (project_root / "dockerfile").read_text(encoding="utf-8")
    required_packages = _copy_runtime_package_roots(first_party_modules)
    assert required_packages, "expected at least one COPY'd Streamlit runtime package"

    missing = [
        package
        for package in sorted(required_packages)
        if not re.search(rf"^COPY\s+{re.escape(package)}/", dockerfile, flags=re.MULTILINE)
    ]
    assert not missing, (
        "dockerfile is missing COPY lines for Streamlit runtime packages that pages import: "
        f"{missing}. Add e.g. `COPY {missing[0]}/ ./{missing[0]}/`."
    )


@pytest.mark.parametrize(
    "module_name",
    sorted(_first_party_imports(_project_root())),
)
def test_streamlit_first_party_imports_resolve(module_name: str) -> None:
    importlib.import_module(module_name)


def test_chess_board_component_frontend_assets_exist(project_root: Path) -> None:
    frontend = project_root / "streamlit_components" / "chess_board" / "frontend"
    assert (frontend / "index.html").is_file()
    for asset in (
        "vendor/chessground.min.js",
        "vendor/chessground.base.css",
        "vendor/chess.min.js",
    ):
        assert (frontend / asset).is_file(), f"missing chess board asset: {asset}"
