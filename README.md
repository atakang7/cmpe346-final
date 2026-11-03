# cmpe346-final

For CMPE 346 students reviewing text classification experiments, this repository holds three classifiers, a LaTeX report, and Beamer slides.

```text
+----------------+  PNG  +--------+
| report/figures | ----> | report |
+----------------+       +--------+
       |
       | PNG
       v
+--------------+
| presentation |
+--------------+
```

## Quick start

With Python 3.11, from the repository root:

```bash
python -m pip install -r student1_atakan_media_bias/requirements.txt
python student1_atakan_media_bias/main.py
```

## Layout

| Path | What it is |
| --- | --- |
| [student1_atakan_media_bias](student1_atakan_media_bias/README.md) | BABE media bias classification |
| [student2_ece_sexism](student2_ece_sexism/README.md) | EDOS sexism classification |
| [student3_talha_ghost_job_classifier](student3_talha_ghost_job_classifier/README.md) | Job posting ghost risk classification |
| [report](report/README.md) | LaTeX report and shared figures |
| [presentation](presentation/README.md) | Beamer slides |

## Docs

- [Documentation index](docs/README.md)
- [Contributing](CONTRIBUTING.md)
- [Build the report and slides](docs/how-to/build-submission.md)
- [Data and output formats](docs/reference/data-and-outputs.md)
- [Architecture](docs/explanation/architecture.md)

## Limits

- The EDOS input CSV must be obtained separately.
- Training downloads model weights and datasets; it needs network access.
- CI checks source syntax. Full model training and TeX rendering run locally.
