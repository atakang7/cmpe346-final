Part of [cmpe346-final](../README.md)

# Report

LaTeX source for the CMPE346 group report and its saved result figures.

## Run

From this directory with a TeX installation containing `pdflatex`:

```bash
bash render.sh
```

## Test

```bash
bash -n render.sh
```

## Config

| Var | Default | What |
| --- | --- | --- |
| Entry file | `template346.tex` | Group report source |
| Style | `styles/ibu-thesis.sty` | Local thesis style |

## Interfaces

| Path | Reads or writes |
| --- | --- |
| `template346.tex` | Reads report text and figure references |
| `figures/*.png` | Saved figures used by report and slides |
| `2_cmpe346_final.pdf` | PDF written by `render.sh` |
| `guide.tex` | Thesis template example, separate from the report build |
| `sample.bib` | Template bibliography, separate from the report build |

See [build the submission](../docs/how-to/build-submission.md).
