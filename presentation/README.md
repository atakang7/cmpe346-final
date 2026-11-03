Part of [cmpe346-final](../README.md)

# Presentation

Beamer slides for the CMPE346 text classification results.

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
| Entry file | `presentation.tex` | Beamer source |
| Figure path | `../report/figures/` | Shared result images |

## Interfaces

| Path | Reads or writes |
| --- | --- |
| `presentation.tex` | Reads slide text and report figures |
| `2_cmpe346_final_presentation.pdf` | PDF written by `render.sh` |

See [build the submission](../docs/how-to/build-submission.md).
