#!/usr/bin/env bash
declare -A models=(
  ["nomic-ai/nomic-embed-text-v1-unsupervised"]="bert-base-uncased"
  ["nomic-ai/nomic-embed-text-v1"]="bert-base-uncased"
  ["BASF-AI/ChEmbed-vanilla"]="bert-base-uncased"
  ["BASF-AI/ChEmbed-full"]="BASF-AI/ChemVocab"
  ["BASF-AI/ChEmbed-plug"]="BASF-AI/ChemVocab"
  ["BASF-AI/ChEmbed-prog"]="BASF-AI/ChemVocab"
)

for model in "${!models[@]}"; do
  tokenizer="${models[$model]}"
  echo "Evaluating model=$model  tokenizer=$tokenizer"

  python benchmark/eval_nomic.py \
    -cp "$model" \
    --benchmark chemrxiv \
    -op benchmark/nc2-2 \
    --tokenizer_name "$tokenizer" \
    --seq_length 2048

  python benchmark/eval_nomic.py \
    -cp "$model" \
    --benchmark chemteb \
    -op benchmark/chemteb \
    --tokenizer_name "$tokenizer" \
    --seq_length 2048

  python benchmark/eval_nomic.py \
    -cp "$model" \
    --benchmark mteb-v2 \
    -op benchmark/mteb-v2 \
    --tokenizer_name "$tokenizer" \
    --seq_length 2048
done
