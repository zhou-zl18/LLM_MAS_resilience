vllm serve ../Qwen2.5-7B-Instruct/ \
    --enable-lora \
    --lora-modules mylora=../GRPO_output/GRPO_result/checkpoint-795/