---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- geoai
- outdoor-assistant
- hackathon
- slm
- qwen2.5
base_model: Qwen/Qwen2.5-3B-Instruct
---

# GeoAI Outdoor Assistant — SLM Configuration

This repository contains the configuration and system prompt for the **GeoAI Outdoor Assistant**, an AI-powered outdoor activity recommender grounded in live weather, air quality, and geospatial context.

## Base Model

- **Model**: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- **Parameters**: 2.7B (within the ≤3B limit)
- **Architecture**: Qwen2.5 (Transformer-based causal LM)
- **License**: Apache 2.0
- **Quantization**: 4-bit NF4 via bitsandbytes (at inference time)

## How We Use the Model

The SLM is used **exclusively for natural language response generation**. It does NOT:
- Recall weather data from training
- Generate or hallucinate places/locations
- Make up AQI or temperature values

Instead, the model receives a **structured context** containing pre-scored, live-fetched data (weather, AQI, geospatial analysis, daylight info, travel times) and generates a human-readable recommendation grounded in that data.

### System Prompt

The model operates with a system prompt that instructs it to:
1. Give clear, specific recommendations (which place, when)
2. Explain WHY based on actual data values
3. Provide practical advice (what to carry, precautions)
4. Mention when to AVOID and why
5. Account for personal health conditions
6. Never invent data — only use what's provided

## Intended Use

This model configuration is designed for:
- Answering free-form questions about outdoor activities
- Generating recommendations for 10+ activities (running, cycling, cricket, golf, tennis, hiking, picnic, outdoor events, photography, football)
- Personalizing advice for health conditions (asthma, elderly, children, pregnancy)
- Indian geography and weather context (primary focus)

## Data Pipeline

```
User Query → Query Parser → Live Data Fetch → Scoring Engine → SLM (this model)
                                  ↓
                    Weather (Open-Meteo) + AQI (WAQI)
                    + Places (OSM) + Daylight + Routing
```

## Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype="float16",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
```

## Limitations

- The model may occasionally produce less fluent text compared to larger models
- Response quality depends on the quality of live data fetched
- Optimized for Indian cities/regions but works globally
- 4-bit quantization may slightly reduce response quality vs. full precision

## Citation

GeoAI Outdoor Assistant — October 2026.
