# Architecture

The repository contains three independent text classification experiments. Each has its own dataset loader, training loop, model definitions, and metrics. They use a bidirectional LSTM, BERT, and RoBERTa with task-specific data.

The folders stay separate because their inputs and labels differ. BABE supplies train and test splits. EDOS reads an aggregated CSV with named splits. The ghost-job loader downloads a parquet file, assigns labels from visible posting phrases, and creates grouped train, validation, and test splits. This repeats some model code across folders.

The report and Beamer slides are LaTeX documents. Their tracked figures live in `report/figures/`; they are copies of saved experiment outputs. Rendering does not retrain the models.

The ghost-job label reflects phrases such as pending awards and talent-pool language. It cannot establish whether an employer filled a position. Each task's score measures its own labels, data split, and F1 averaging method.
