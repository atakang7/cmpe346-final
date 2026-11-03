#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error template346.tex
pdflatex -interaction=nonstopmode -halt-on-error template346.tex
rm -f -- *.aux *.log *.out *.toc *.fls *.fdb_latexmk
mv template346.pdf 2_cmpe346_final.pdf
printf 'Created 2_cmpe346_final.pdf\n'
