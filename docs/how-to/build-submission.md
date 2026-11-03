# Build the report and slides

1. Install a TeX distribution with `pdflatex` and the packages referenced by `report/template346.tex` and `presentation/presentation.tex`.
2. From the repository root, render both sources:

```bash
bash report/render.sh
bash presentation/render.sh
```

3. Read the generated PDFs at `report/2_cmpe346_final.pdf` and `presentation/2_cmpe346_final_presentation.pdf`.

Both builds use the committed images in `report/figures/`. A failed TeX command stops the corresponding script.
