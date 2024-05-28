#!/bin/bas
# python scripts/convert_sqa_to_llava.py \
#     convert_to_llava \
#     --base-dir /path/to/ScienceQA/data/scienceqa \
#     --prompt-format "CQM-A" \
#     --split test

python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/llava-v1.5-13b-lora-vqvae  \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_result.json
