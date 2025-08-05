# LLM_MAS_resilience

The official PyTorch implementation of "ResMAS: Resilience Optimization in LLM-based Multi-agent Systems".

## Usage

### Stage 0: Preparation
- Download Qwen2.5-7B-Instruct from https://www.modelscope.cn/models/qwen/Qwen2.5-7B-Instruct, and put it into the "Qwen2.5-7B-Instruct" folder.
- Install LLaMA-Factory from https://github.com/hiyouga/LLaMA-Factory
- Set API keys in llm_configs.py
- Use the ``pip install -r requirements.txt`` command to install packages used in this project

### Stage 1: Topology Optimization

**1. Supervised fine-tuning**
```
cd sft
bash run_sft.sh
```

**2. Training reward model**
```
cd reward_model
bash run.sh
```

**3. GRPO training**
```
cd GRPO
bash run.sh
```

**4. Generate MAS topology**
```
cd GRPO
bash run_api.sh
python generate_topology.py
```


### Stage 2: Prompt Optimization

```
cd prompt_optimization
MATH: python run_math_optim.py
MMLU: python run_mmlu_optim.py
Chess: python run_chess_optim.py
```

### Stage 3: Evaluation
```
cd evaluation
bash run.sh
```



