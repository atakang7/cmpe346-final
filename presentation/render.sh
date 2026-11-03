#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error presentation.tex
pdflatex -interaction=nonstopmode -halt-on-error presentation.tex
rm -f -- *.aux *.log *.out *.nav *.snm *.toc *.fls *.fdb_latexmk
mv presentation.pdf 2_cmpe346_final_presentation.pdf
printf 'Created 2_cmpe346_final_presentation.pdf\n'
