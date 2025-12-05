# Reproducing Preference Alignment on Anthropic HH with a Small GPT

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CS329H: Machine Learning from Human Preferences - Final Project**  
*Authors:* Ishan Pakuwal and Tanaya Yadav  
*Institution:* Stanford University  
*Date:* December 2025

## 📖 Overview

This repository contains a complete, reproducible implementation of preference alignment for language models using:
- **Supervised Fine-Tuning (SFT)** on preferred responses from the Anthropic HH dataset
- **Direct Preference Optimization (DPO)** for aligning with human preferences
- **Comprehensive evaluation** across 5 metrics: perplexity, toxicity, refusal accuracy, human preferences, and response quality

We train an 85M-parameter GPT-style model on a single GPU in under 3 hours, achieving:
- **68% win rate** in pairwise human preferences vs. SFT baseline
- **45% reduction** in toxicity (mean score: 0.089 → 0.049)
- **18-point improvement** in refusal accuracy (74% → 92%)

## 🎯 Key Features

- ✅ **Single-GPU Training**: Runs on free Google Colab T4 GPU
- ✅ **Fast Iteration**: Complete pipeline in 2-3 hours
- ✅ **Minimal Dependencies**: Based on nanoGPT for clarity
- ✅ **Comprehensive Evaluation**: 5 complementary metrics
- ✅ **Fully Reproducible**: Fixed seeds, exact package versions
- ✅ **Educational**: Designed for learning preference alignment

## 📂 Repository Structure

```
cs329h-preference-alignment/
├── CS329H_Complete_Project.ipynb    # Main notebook with complete pipeline
├── README.md                         # This file
├── requirements.txt                  # Python dependencies with versions
├── verify_setup.py                   # Environment verification script
├── EVALUATION_METHODOLOGY.md         # Detailed evaluation protocols
├── results/                          # Generated outputs (created at runtime)
│   ├── sft_final_model.pt           # Supervised fine-tuned model checkpoint
│   ├── dpo_final_model.pt           # DPO-optimized model checkpoint
│   ├── best_sft_model.pt            # Best SFT checkpoint (early stopping)
│   ├── training_curves.png          # Training dynamics visualization
│   ├── toxicity_distribution.png    # Toxicity scores comparison
│   ├── model_comparison.csv         # Quantitative metrics table
│   └── results_summary.json         # Complete results in JSON format
├── annotations/                      # Human evaluation data (created at runtime)
│   ├── pairwise_preferences_for_annotation.csv
│   └── helpfulness_for_annotation.csv
└── paper/                            # Final manuscript
    ├── manuscript.tex
    ├── refs.bib
    └── manuscript.pdf
