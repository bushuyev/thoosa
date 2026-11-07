# HBAC workflow

The HBAC sample project uses EEG data from the [HMS – Harmful Brain Activity Classification competition on Kaggle](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/data). It demonstrates a multi-stage time-series classification pipeline, from discovering recurring signal patterns to training a final classifier.

[`run_all`](../crates/thoosa_projects/thoosa_hbac/src/run_all.rs) coordinates the following steps:

1. **Find and select motifs.** [`FindMotifsStage`](../crates/thoosa_motifs/src/stages/find_motifs.rs) compares EEG time-series windows using Euclidean distance to discover recurring patterns, or motifs. [`SelectMotifsStage`](../crates/thoosa_motifs/src/stages/select_motifs.rs) evaluates the candidates against configured quality criteria and retains suitable motifs.
2. **Train motif networks.** [`TrainMotifNets`](../crates/thoosa_motifs/src/stages/train_motif_nets.rs) trains a separate small neural network for each selected motif. These networks learn to recognize their corresponding patterns in EEG windows.
3. **Train label networks.** [`TrainLabelNets`](../crates/thoosa_motifs/src/stages/train_label_nets.rs) trains label-specific neural networks using the motif networks' outputs as features, connecting detected patterns to brain-activity labels.
4. **Train the final classifier.** [`TrainFinalNet`](../crates/thoosa_motifs/src/stages/train_final_net.rs) learns to combine the label networks' outputs into the final classification predictions.

The workflow passes saved run-configuration IDs between stages so downstream training uses the preceding stages' results. It supports grid search and [parameter optimization](opt.md), with optional plots for inspecting selected motifs.

Exploratory runs with a few hours of training have produced modest classification accuracy, above random guessing. The project's primary goal is to demonstrate a pipeline with varied computational tasks—signal comparison, motif selection, and several levels of neural-network training—using Thoosa's asynchronous execution and memory-pressure handling, rather than to provide a competitive classification model.

For command examples covering data import, plotting, individual stages, and the complete `run-all` workflow, see the [HBAC command cheat sheet](../crates/thoosa_projects/thoosa_hbac/commands.txt). Run Cargo commands from the repository's `crates/` directory and adapt workspace paths, resource groups, and run-configuration IDs to your data.
