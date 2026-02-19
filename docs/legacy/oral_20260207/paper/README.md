# ProveTok Paper (NeurIPS-style LaTeX)

Entry: `paper/main.tex`

Notes:
- Figures are referenced from `../docs/paper_assets/figures/`.
- BibTeX is reused from `../docs/paper_assets/references.bib`.
- Tables are LaTeX conversions of `../docs/paper_assets/tables/*.md`.

## Build (local)
```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

If `latexmk` is unavailable, use `pdflatex` + `bibtex` manually:
```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Refresh figures/tables
```bash
python scripts/paper/build_readme_figures.py --out-dir docs/paper_assets/figures
python scripts/paper/build_readme_tables.py --out-dir docs/paper_assets/tables
```

## Proof gate (claim-level audit)
```bash
python scripts/proof_check.py --profile real
python scripts/oral_audit.py --sync --out outputs/oral_audit.json --strict
```

