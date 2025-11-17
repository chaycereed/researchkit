import argparse
import json
import sys
import platform
from pathlib import Path
import textwrap

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

# Simple ANSI styles for nicer CLI output
RESET = "\033[0m"
BOLD = "\033[1m"

CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
DIM = "\033[2m"


# -----------------------
# Core project generation
# -----------------------

def create_base_structure(root: Path) -> None:
    """
    Create the core folder structure for a study.
    """
    # Core directories
    (root / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (root / "data" / "processed").mkdir(parents=True, exist_ok=True)
    (root / "data" / "external").mkdir(parents=True, exist_ok=True)

    (root / "metadata").mkdir(parents=True, exist_ok=True)

    (root / "analysis" / "notebooks").mkdir(parents=True, exist_ok=True)
    (root / "analysis" / "scripts").mkdir(parents=True, exist_ok=True)

    (root / "docs").mkdir(parents=True, exist_ok=True)

    (root / "results" / "figures").mkdir(parents=True, exist_ok=True)
    (root / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (root / "results" / "reports").mkdir(parents=True, exist_ok=True)

    (root / "config").mkdir(parents=True, exist_ok=True)


def create_notebook_file(path: Path, cells) -> None:
    """
    Generic helper to write a Jupyter notebook with the given cells.
    """
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=2), encoding="utf-8")


def create_qc_notebook(path: Path) -> None:
    """
    Create a QC notebook with prepped code cells and a fixed seed.
    """
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Quality Control (QC) Template\n",
                "\n",
                "This notebook provides a starting point for basic QC checks:\n",
                "\n",
                "- Load raw data\n",
                "- Inspect structure and missingness\n",
                "- Look at basic distributions\n",
                "- Flag obvious data issues\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from pathlib import Path\n",
                "import random\n",
                "\n",
                "RANDOM_SEED = 42\n",
                "np.random.seed(RANDOM_SEED)\n",
                "random.seed(RANDOM_SEED)\n",
                "\n",
                "DATA_DIR = Path('data') / 'raw'\n",
                "# TODO: replace with your actual raw data file\n",
                "raw_path = DATA_DIR / 'your_raw_data.csv'\n",
                "\n",
                "# df = pd.read_csv(raw_path)\n",
                "# df.head()\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Basic structure and summary\n",
                "# Uncomment after loading df\n",
                "# df.info()\n",
                "# df.describe(include='all')\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Missingness patterns\n",
                "# Uncomment after loading df\n",
                "# missing_frac = df.isna().mean().sort_values(ascending=False)\n",
                "# missing_frac.head(20)\n",
                "\n",
                "# plt.figure(figsize=(8, 4))\n",
                "# missing_frac.plot(kind='bar')\n",
                "# plt.ylabel('Proportion missing')\n",
                "# plt.tight_layout()\n",
                "# plt.show()\n",
            ],
        },
    ]
    create_notebook_file(path, cells)


def create_cleaning_notebook(path: Path) -> None:
    """
    Create a cleaning notebook with prepped code cells and a fixed seed.
    """
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Data Cleaning Template\n",
                "\n",
                "This notebook is for transforming raw data into analysis-ready datasets.\n",
                "\n",
                "Typical tasks:\n",
                "- Rename columns\n",
                "- Recode categorical variables\n",
                "- Handle missing values\n",
                "- Merge data sources\n",
                "- Create derived variables\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "from pathlib import Path\n",
                "import random\n",
                "\n",
                "RANDOM_SEED = 42\n",
                "np.random.seed(RANDOM_SEED)\n",
                "random.seed(RANDOM_SEED)\n",
                "\n",
                "RAW_DIR = Path('data') / 'raw'\n",
                "PROCESSED_DIR = Path('data') / 'processed'\n",
                "\n",
                "# TODO: replace with your actual raw data file\n",
                "raw_path = RAW_DIR / 'your_raw_data.csv'\n",
                "\n",
                "# df = pd.read_csv(raw_path)\n",
                "# df.head()\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Column renaming example\n",
                "# rename_map = {\n",
                "#     'old_name': 'new_name',\n",
                "# }\n",
                "# df = df.rename(columns=rename_map)\n",
                "\n",
                "# Categorical recoding example\n",
                "# df['some_category'] = df['some_category'].replace({\n",
                "#     'Yes': 1,\n",
                "#     'No': 0,\n",
                "# })\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Missing data handling examples\n",
                "# Option 1: drop rows with any missing values in key variables\n",
                "# key_vars = ['var1', 'var2']\n",
                "# df = df.dropna(subset=key_vars)\n",
                "\n",
                "# Option 2: simple imputation\n",
                "# df['some_numeric'] = df['some_numeric'].fillna(df['some_numeric'].mean())\n",
                "\n",
                "# Add any derived variables here\n",
                "# df['bmi'] = df['weight_kg'] / (df['height_m'] ** 2)\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Save cleaned data\n",
                "PROCESSED_DIR.mkdir(parents=True, exist_ok=True)\n",
                "# cleaned_path = PROCESSED_DIR / 'your_cleaned_data.csv'\n",
                "# df.to_csv(cleaned_path, index=False)\n",
                "\n",
                "# cleaned_path\n",
            ],
        },
    ]
    create_notebook_file(path, cells)


def create_analysis_notebook(path: Path) -> None:
    """
    Create an analysis notebook with prepped code cells and a fixed seed.
    """
    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Analysis Template\n",
                "\n",
                "Use this notebook for your primary analyses.\n",
                "\n",
                "Typical steps:\n",
                "- Load cleaned data\n",
                "- Run descriptive statistics\n",
                "- Fit models\n",
                "- Create figures and tables\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from pathlib import Path\n",
                "import random\n",
                "\n",
                "RANDOM_SEED = 42\n",
                "np.random.seed(RANDOM_SEED)\n",
                "random.seed(RANDOM_SEED)\n",
                "\n",
                "PROCESSED_DIR = Path('data') / 'processed'\n",
                "\n",
                "# TODO: replace with your actual cleaned data file\n",
                "clean_path = PROCESSED_DIR / 'your_cleaned_data.csv'\n",
                "\n",
                "# df = pd.read_csv(clean_path)\n",
                "# df.head()\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Descriptive statistics\n",
                "# df.describe(include='all')\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Example: simple group comparison or regression\n",
                "# import statsmodels.formula.api as smf\n",
                "\n",
                "# model = smf.ols('outcome ~ predictor1 + predictor2', data=df).fit()\n",
                "# model.summary()\n",
            ],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": [
                "# Example: figure placeholder\n",
                "# plt.figure(figsize=(6, 4))\n",
                "# df['some_variable'].hist(bins=30)\n",
                "# plt.xlabel('some_variable')\n",
                "# plt.ylabel('Count')\n",
                "# plt.tight_layout()\n",
                "# plt.show()\n",
            ],
        },
    ]
    create_notebook_file(path, cells)


def generate_project(study_name: str, description: str, options: dict) -> None:
    """
    Create the full study scaffold based on the selected options.
    """
    root = Path(study_name)

    if root.exists():
        print(f"{YELLOW}⚠ Directory '{study_name}' already exists. Files may be overwritten.{RESET}")

    root.mkdir(parents=True, exist_ok=True)

    # 1) Folder structure
    create_base_structure(root)

    # 2) Top-level README
    readme_text = textwrap.dedent(f"""
    # {study_name}

    {description}

    ---

    ## 1. Study Overview

    Use this section to briefly describe:
    - The main research question(s)
    - The population and setting
    - The primary outcomes
    - The key predictors or exposures

    ## 2. Folder Structure

    - `data/`
      - `raw/` – original, unmodified source data (read-only)
      - `processed/` – cleaned, analysis-ready datasets
      - `external/` – external reference data (e.g., census, public datasets)
    - `metadata/` – study metadata, data dictionary, and codebook
    - `analysis/`
      - `notebooks/` – QC, cleaning, and analysis notebooks
      - `scripts/` – reusable analysis/utility scripts
    - `docs/` – study-level documentation and reproducibility materials
    - `results/` – figures, tables, and reports
    - `config/` – configuration files (e.g., environment, credentials templates)

    ## 3. Naming Conventions

    Recommended conventions:
    - Use `snake_case` for file and variable names.
    - Prefer descriptive names (e.g., `sleep_quality_score` instead of `sq`).
    - Store **only** unmodified source data in `data/raw/`.
    - Save all cleaned, transformed data to `data/processed/`.
    - If you create multiple versions of a dataset, use suffixes like:
      - `_v1`, `_v2`, or
      - `_clean`, `_analysis`, `_derived`.

    ## 4. Getting Started

    1. Fill in `metadata/study_metadata.yml`.
    2. Define variables in `metadata/data_dictionary.csv`.
    3. Flesh out `metadata/codebook.md` (if generated).
    4. Use the QC and cleaning notebooks to explore and clean the data.
    5. Use the analysis notebook for your main models and results.

    ## 5. Reproducibility

    - Keep all raw data immutable.
    - Document all transformations in notebooks or scripts.
    - Use the reproducibility checklist (if generated) as a guide before sharing,
      submitting, or archiving this project.
    """).strip() + "\n"

    (root / "README.md").write_text(readme_text, encoding="utf-8")

    # 3) Metadata file (YAML)
    metadata_text = textwrap.dedent(f"""
    # Study metadata for {study_name}
    # Fill in the fields below to document your study.
    # This file is YAML-formatted; keep indentation consistent.

    title: "{study_name}"
    short_title: ""
    description: "{description or ''}"

    principal_investigator:
      name: ""
      institution: ""
      email: ""

    co_investigators:
      - name: ""
        role: ""
        email: ""

    study_type: ""  # e.g., observational, RCT, longitudinal, cross-sectional
    population: ""  # e.g., adults 18–35 in [region]
    recruitment_method: ""  # e.g., online ads, clinic recruitment
    setting: ""  # e.g., online, in-person lab, clinic

    start_date: ""
    end_date: ""

    sample_size:
      planned: ""
      final: ""

    primary_outcomes:
      - name: ""
        description: ""
    secondary_outcomes:
      - name: ""
        description: ""

    measures:
      - name: ""
        domain: ""  # e.g., sleep, mood, stress
        instrument: ""  # e.g., PSQI, PHQ-9
        reference: ""  # citation or URL

    data_sources:
      - name: ""
        type: ""  # e.g., survey, wearable, EHR, registry
        description: ""
        frequency: ""  # e.g., daily, baseline-only, weekly

    ethics:
      irb_protocol_id: ""
      consent_required: true
      notes: ""
    """).lstrip()

    (root / "metadata" / "study_metadata.yml").write_text(metadata_text, encoding="utf-8")

    # 4) Data dictionary skeleton (CSV)
    data_dict_text = textwrap.dedent("""
    variable_name,label,type,allowed_values,source,notes
    participant_id,Unique participant ID,string,,survey,Stable ID used to link records across tables
    """).lstrip()

    (root / "metadata" / "data_dictionary.csv").write_text(data_dict_text, encoding="utf-8")

    # 5) .gitignore
    gitignore_text = textwrap.dedent("""
    # Python
    __pycache__/
    *.py[cod]
    .Python
    env/
    venv/
    .venv/

    # Jupyter
    .ipynb_checkpoints/

    # Data
    data/raw/*
    data/processed/*
    data/external/*

    # OS
    .DS_Store
    """).lstrip()

    (root / ".gitignore").write_text(gitignore_text, encoding="utf-8")

    # Optional components
    if options.get("codebook"):
        codebook_text = textwrap.dedent("""
        # Codebook

        Use this document to describe each measure, scale, and derived variable in detail.
        It complements the `data_dictionary.csv` file.

        ## 1. Measures

        ### Example measure

        - **Name:**  
        - **Domain:** (e.g., sleep, mood, stress)
        - **Instrument/Scale:**  
        - **Items:**  
        - **Time frame:** (e.g., past week, past 2 weeks)
        - **Response format:** (e.g., 1–5 Likert, yes/no)
        - **Scoring rules:** (sum, mean, reverse-coded items, etc.)
        - **Interpretation:** (what do higher/lower scores mean?)
        - **Reference / citation:**  

        Repeat this section for each major measure.

        ## 2. Derived Variables

        ### Example derived variable

        - **Name:**  
        - **Based on variables:**  
        - **Computation:**  
        - **Rationale:**  
        - **Notes:**  

        ## 3. Missing Data Rules

        Describe how you will handle missing data for key variables:
        - Exclusion rules
        - Imputation strategies (if any)
        - Flags for missingness or quality
        """).lstrip()
        (root / "metadata" / "codebook.md").write_text(codebook_text, encoding="utf-8")

    if options.get("qc"):
        create_qc_notebook(
            root / "analysis" / "notebooks" / "01_qc_template.ipynb",
        )

    if options.get("clean"):
        create_cleaning_notebook(
            root / "analysis" / "notebooks" / "02_cleaning_template.ipynb",
        )

    if options.get("analysis"):
        create_analysis_notebook(
            root / "analysis" / "notebooks" / "03_analysis_template.ipynb",
        )

    if options.get("checklist"):
        checklist_text = textwrap.dedent("""
        # Reproducibility Checklist

        Use this checklist to ensure your study can be understood and reproduced
        by other researchers (or by you in the future).

        ## Data and Structure

        - [ ] All raw data is stored in `data/raw/` and never modified in-place.
        - [ ] All cleaned data is stored in `data/processed/`.
        - [ ] External reference data is stored in `data/external/`.

        ## Metadata and Documentation

        - [ ] `metadata/study_metadata.yml` is complete and up to date.
        - [ ] `metadata/data_dictionary.csv` describes all variables used in analysis.
        - [ ] `metadata/codebook.md` documents measures and derived variables (if used).
        - [ ] `README.md` reflects the current state of the project.

        ## Analysis

        - [ ] QC notebook documents data quality checks.
        - [ ] Cleaning notebook documents all transformations.
        - [ ] Analysis notebook(s) can be run from top to bottom without errors.
        - [ ] Random seeds are set for any stochastic procedures (if applicable).
        - [ ] All figures and tables used in reports are generated from code.

        ## Environment

        - [ ] Python/R environment and key package versions are recorded.
        - [ ] Any required credentials/config are stored securely (not committed).

        ## Sharing and Archiving

        - [ ] Sensitive data is handled according to IRB/ethics requirements.
        - [ ] A de-identified version of the dataset is prepared (if needed).
        - [ ] A plan exists for long-term storage or sharing of data and code.
        """).lstrip()
        (root / "docs" / "reproducibility_checklist.md").write_text(checklist_text, encoding="utf-8")

    if options.get("cite"):
        citation_text = textwrap.dedent(f"""
        # How to cite this project

        If you use this study or its materials, you may cite it as:

        > [Authors]. ({{YYYY}}). {study_name}. Unpublished study.

        You may also acknowledge **researchkit** as the tool used to scaffold
        the project structure and documentation:

        > This project was initialized using researchkit, a study template
        > generator for public-health and behavioral research.
        """).lstrip()
        (root / "docs" / "CITATION_template.md").write_text(citation_text, encoding="utf-8")


# -----------------------
# Environment info utility
# -----------------------

def gather_env_info():
    """
    Gather basic environment information for reproducibility.
    """
    python_version = sys.version.split()[0]
    plat = platform.platform()

    # Common scientific/DS packages to probe
    packages_to_check = [
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
        "statsmodels",
        "sklearn",
        "seaborn",
        "polars",
        "torch",
        "tensorflow",
    ]

    pkg_versions = {}
    for pkg in packages_to_check:
        try:
            pkg_versions[pkg] = version(pkg)
        except PackageNotFoundError:
            pkg_versions[pkg] = "not installed"

    return {
        "python_version": python_version,
        "platform": plat,
        "packages": pkg_versions,
    }


def print_env_info(info: dict) -> None:
    """
    Pretty-print environment info to the terminal.
    """
    print(f"{BOLD}{CYAN}Environment information{RESET}\n")
    print(f"{MAGENTA}Python version:{RESET} {info['python_version']}")
    print(f"{MAGENTA}Platform:{RESET} {info['platform']}\n")

    print(f"{BOLD}Common packages:{RESET}")
    for name, ver in info["packages"].items():
        status = ver if ver != "not installed" else f"{YELLOW}not installed{RESET}"
        print(f"  - {name}: {status}")
    print()


def save_env_info_markdown(info: dict, path: Path) -> None:
    """
    Save environment info to a Markdown file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Environment Information\n",
        "\n",
        f"- Python: {info['python_version']}\n",
        f"- Platform: {info['platform']}\n",
        "\n",
        "## Python packages\n",
        "\n",
    ]

    for name, ver in info["packages"].items():
        lines.append(f"- **{name}**: {ver}\n")

    path.write_text("".join(lines), encoding="utf-8")


def run_info_command() -> None:
    """
    Run the `researchkit info` subcommand.
    """
    info = gather_env_info()
    print_env_info(info)

    # Ask whether to save into docs/environment_info.md in current directory
    cwd = Path.cwd()
    default_path = cwd / "docs" / "environment_info.md"

    answer = input(
        f"{BOLD}Save this information to{RESET} {default_path}? (Y/n): "
    ).strip().lower() or "y"

    if answer.startswith("y"):
        save_env_info_markdown(info, default_path)
        print(f"{GREEN}✔ Saved environment info to {default_path}{RESET}")
    else:
        print(f"{DIM}Skipped saving environment info.{RESET}")


# -----------------------
# CLI interaction
# -----------------------

def interactive_init() -> None:
    print(f"{BOLD}{CYAN}🔬 Welcome to researchkit – Study Template Generator{RESET}\n")

    study_name = input(f"{BOLD}Study name{RESET} (e.g., sleep_mood_2025): ").strip()
    if not study_name:
        print(f"{YELLOW}⚠ Study name is required. Exiting.{RESET}")
        return

    description = input(f"{BOLD}Short description{RESET} (optional): ").strip()

    print(f"\n{BOLD}Choose a template profile:{RESET}")
    print(f"  {CYAN}1){RESET} Minimal  – core structure + metadata + data dictionary")
    print(f"  {CYAN}2){RESET} Standard – minimal + codebook + QC + cleaning + analysis")
    print(f"  {CYAN}3){RESET} Full     – standard + reproducibility checklist + citation")
    print(f"  {CYAN}4){RESET} Custom   – choose components one by one")

    profile = input(f"{BOLD}Enter choice [1-4]{RESET} (default 2): ").strip() or "2"

    options = {
        "codebook": False,
        "qc": False,
        "clean": False,
        "analysis": False,
        "checklist": False,
        "cite": False,
    }

    if profile == "1":  # Minimal
        pass  # extras remain False

    elif profile == "2":  # Standard
        options.update({
            "codebook": True,
            "qc": True,
            "clean": True,
            "analysis": True,
        })

    elif profile == "3":  # Full
        options.update({
            "codebook": True,
            "qc": True,
            "clean": True,
            "analysis": True,
            "checklist": True,
            "cite": True,
        })

    elif profile == "4":  # Custom
        print()
        include_codebook = input(f"Include {BOLD}codebook template{RESET}? (Y/n): ").strip().lower() or "y"
        include_qc = input(f"Include {BOLD}QC notebook{RESET}? (Y/n): ").strip().lower() or "y"
        include_clean = input(f"Include {BOLD}cleaning notebook{RESET}? (Y/n): ").strip().lower() or "y"
        include_analysis = input(f"Include {BOLD}analysis notebook{RESET}? (Y/n): ").strip().lower() or "y"
        include_checklist = input(f"Include {BOLD}reproducibility checklist{RESET}? (Y/n): ").strip().lower() or "y"
        include_citation = input(f"Include {BOLD}citation template{RESET}? (Y/n): ").strip().lower() or "y"

        options.update({
            "codebook": include_codebook.startswith("y"),
            "qc": include_qc.startswith("y"),
            "clean": include_clean.startswith("y"),
            "analysis": include_analysis.startswith("y"),
            "checklist": include_checklist.startswith("y"),
            "cite": include_citation.startswith("y"),
        })
    else:
        print(f"{YELLOW}⚠ Invalid profile selection. Using Standard profile.{RESET}")
        options.update({
            "codebook": True,
            "qc": True,
            "clean": True,
            "analysis": True,
        })

    print(f"\n{BOLD}Creating study scaffold...{RESET}")
    generate_project(study_name, description, options)
    print(f"{GREEN}✔ Done! Created study at ./{study_name}{RESET}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="researchkit",
        description="Research Kit - Automated setup for folders, notebooks, and READMEs for reproducible research.",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("init", help="Initialize a new study project")
    subparsers.add_parser("info", help="Show and optionally save environment information")

    args = parser.parse_args()

    if args.command == "init":
        interactive_init()
    elif args.command == "info":
        run_info_command()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()