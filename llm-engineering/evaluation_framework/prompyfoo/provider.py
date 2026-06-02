# provider.py
import sys
import os

# Replicate the model simulation from eval_runner.py
SIMULATED_MODELS = {
    "gpt-4o": lambda inp: f"Based on the question about {inp.split()[0:3]}, the answer involves careful analysis of the key factors. The primary consideration is relevance to the topic at hand, with supporting evidence from established sources.",
    "baseline-v1": lambda inp: f"The answer to your question about {' '.join(inp.split()[0:5])} is as follows: this topic requires understanding of multiple interconnected concepts.",
    "baseline-v2": lambda inp: f"Regarding {' '.join(inp.split()[0:4])}: the short answer is that it depends on context, but here are the key points you should consider for a complete understanding.",
}

def call_api(prompt, options, context):
    model_name = options.get('config', {}).get('model', 'gpt-4o')
    generator = SIMULATED_MODELS.get(model_name)
    if not generator:
        return {'error': f"Unknown model: {model_name}"}
    
    # Generate the simulated response
    response_text = generator(prompt)
    return {'output': response_text}
