# researchkit  
Automated setup for folders, notebooks, and READMEs for reproducible research

`researchkit` is a small Python toolkit that bootstraps a **reproducible study
template** in a single command. It’s designed for public-health, behavioral, and
clinical researchers who want clean structure, metadata, and analysis scaffolds
from day one.

---

## Features

When you run:

```bash
researchkit init
```

you get:

- **Structured folders** for data, analysis, metadata, docs, and results  
- A **top-level README** describing the study and folder layout  
- A **study metadata YAML** file (`metadata/study_metadata.yml`)  
- A **data dictionary skeleton** (`metadata/data_dictionary.csv`)  
- Optional extras:
  - `metadata/codebook.md` – codebook template for measures & derived variables  
  - QC, cleaning, and analysis **Jupyter notebooks** with:
    - imports and paths pre-wired
    - random seeds pre-set
    - commented example code blocks
  - `docs/reproducibility_checklist.md` – reproducibility checklist  
  - `docs/CITATION.md` – citation/acknowledgment template  
- A useful `.gitignore` for Python, Jupyter, and data files

`researchkit info` also provides a tiny **environment snapshot** utility so you
can record Python and package versions for reproducibility.

---

## Installation

Install directly from GitHub:

```bash
python3 -m pip install git+https://github.com/chaycereed/researchkit.git
```

After installation:

```bash
researchkit --help
```

---

## Usage

### 1. Initialize a new study

```bash
researchkit init
```

You’ll be guided through an interactive wizard:

- Study name  
- Short description  
- Template profile  
  - Minimal  
  - Standard  
  - Full  
  - Custom  

This creates a folder structure like:

```
your_study/
  data/
    raw/
    processed/
    external/
  metadata/
    study_metadata.yml
    data_dictionary.csv
    (codebook.md)
  analysis/
    notebooks/
      01_qc_notebook.ipynb
      02_cleaning_notebook.ipynb
      03_analysis_notebook.ipynb
    scripts/
  docs/
    (reproducibility_checklist.md)
    (CITATION.md)
  results/
    figures/
    tables/
    reports/
  config/
  .gitignore
  README.md
```

---

### 2. Environment snapshot

```bash
researchkit info
```

This prints:

- Python version  
- Platform  
- Versions of common scientific packages  

And optionally saves:

```
docs/environment_info.md
```

---

## Philosophy

`researchkit` is intentionally lightweight:

The goal is simple:  
**make reproducible workflows the default.**

---

## License

MIT License. See `LICENSE` for details.
