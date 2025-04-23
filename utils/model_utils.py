import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any
import yaml
import google.generativeai as genai
import anthropic
import openai
import os


class ModelWrapper:
    def __init__(self, config: Dict[str, Any], api_config: Dict[str, Any] = None):
        self.config = config
        self.api_config = api_config
        self.model = None
        self.tokenizer = None
        self.api_client = None

        if api_config:
            self._init_api_model()
        else:
            self._init_local_model()

    def _init_local_model(self):
        """Initialize local HuggingFace model"""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['name'],
            **self.config['params'],
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['name'], trust_remote_code=True)

    def _init_api_model(self):
        """Initialize API-based model"""
        provider = self.api_config['api_provider'].lower()

        if provider == "gemini":
            genai.configure(api_key=self.api_config['gemini']['api_key'])
            self.api_client = genai.GenerativeModel(
                model_name=self.api_config['gemini']['model_name'],
                safety_settings=self.api_config['gemini'].get('safety_settings', []),
                generation_config={
                    "temperature": self.api_config['gemini']['temperature'],
                    "max_output_tokens": self.api_config['gemini']['max_tokens']
                }
            )
        elif provider == "claude":
            self.api_client = anthropic.Anthropic(
                api_key=self.api_config['claude']['api_key']
            )
        elif provider == "openai":
            self.api_client = openai.Client(
                api_key=self.api_config['openai']['api_key']
            )
        else:
            raise ValueError(f"Unsupported API provider: {provider}")

    def generate_response(self, prompt: str, max_new_tokens: int = 256) -> str:
        """Generate response from model or API"""
        if self.api_client:
            return self._generate_api_response(prompt, max_new_tokens)
        return self._generate_local_response(prompt, max_new_tokens)

    def _generate_local_response(self, prompt: str, max_new_tokens: int) -> str:
        """Generate response from local model"""
        messages = [
            {"role": "system", "content": "You are an expert in natural language processing."},
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _generate_api_response(self, prompt: str, max_new_tokens: int) -> str:
        """Generate response from API"""
        provider = self.api_config['api_provider'].lower()

        if provider == "gemini":
            response = self.api_client.generate_content(
                contents=[{"role": "user", "parts": [{"text": prompt}]}],
                generation_config={
                    "max_output_tokens": min(max_new_tokens, self.api_config['gemini']['max_tokens']),
                    "temperature": self.api_config['gemini']['temperature']
                }
            )
            return response.text

        elif provider == "claude":
            response = self.api_client.messages.create(
                model=self.api_config['claude']['model_name'],
                max_tokens=min(max_new_tokens, self.api_config['claude']['max_tokens']),
                temperature=self.api_config['claude']['temperature'],
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif provider == "openai":
            response = self.api_client.chat.completions.create(
                model=self.api_config['openai']['model_name'],
                max_tokens=min(max_new_tokens, self.api_config['openai']['max_tokens']),
                temperature=self.api_config['openai']['temperature'],
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content

        raise ValueError(f"Unsupported API provider: {provider}")


def load_model(config: Dict[str, Any], api_config: Dict[str, Any] = None):
    """Load either local or API-based model"""
    return ModelWrapper(config, api_config)