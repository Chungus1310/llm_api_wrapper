import os
import time
import random
import requests
from typing import Optional, Dict, Any, List
import google.generativeai as genai

class LLMProviderBase:
    """
    Base class for all LLM providers.
    Subclasses must implement the send_request method to handle provider-specific logic.
    """
    def __init__(self, provider_name: str, api_keys: List[str]):
        self.provider_name = provider_name
        self.api_keys = api_keys  # Round-robin rotation among available keys
        self.key_index = 0

    def get_api_key(self) -> Optional[str]:
        if not self.api_keys:
            return None
        key = self.api_keys[self.key_index]
        # Round-robin increment
        self.key_index = (self.key_index + 1) % len(self.api_keys)
        return key

    def send_request(
        self,
        prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Implement provider-specific request logic and return standardized response.
        Must be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement send_request.")

class MistralProvider(LLMProviderBase):
    def send_request(
        self,
        prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> Dict[str, Any]:
        api_key = self.get_api_key()
        if not api_key:
            return {"error": "No Mistral API key available"}
        # Example request - adapt to actual Mistral API as needed
        try:
            # Placeholder request logic (adjust URL/method)
            response = requests.post(
                "https://api.mistral.ai/generate",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p
                },
                timeout=10
            )
            response.raise_for_status()
            # Standardize response
            return {"provider": "mistral", "response": response.json()}
        except Exception as e:
            return {"error": str(e)}

class AIStudioProvider(LLMProviderBase):
    def send_request(
        self,
        prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> Dict[str, Any]:
        api_key = self.get_api_key()
        if not api_key:
            return {"error": "No AI Studio API key available"}
        # Example request - adapt to actual AI Studio API as needed
        try:
            response = requests.post(
                "https://api.aistudio.ai/generate",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p
                },
                timeout=10
            )
            response.raise_for_status()
            return {"provider": "aistudio", "response": response.json()}
        except Exception as e:
            return {"error": str(e)}

class OpenRouterProvider(LLMProviderBase):
    def send_request(
        self,
        prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> Dict[str, Any]:
        api_key = self.get_api_key()
        if not api_key:
            return {"error": "No OpenRouter API key available"}
        # Example request - adapt to actual OpenRouter API as needed
        try:
            response = requests.post(
                "https://api.openrouter.ai/generate",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p
                },
                timeout=10
            )
            response.raise_for_status()
            return {"provider": "openrouter", "response": response.json()}
        except Exception as e:
            return {"error": str(e)}

class HuggingFaceProvider(LLMProviderBase):
    def send_request(
        self,
        prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> Dict[str, Any]:
        api_key = self.get_api_key()
        if not api_key:
            return {"error": "No Hugging Face API key available"}
        # Example request - adapt to actual Hugging Face Inference API as needed
        try:
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"inputs": prompt, "options": {"temperature": temperature, "top_p": top_p}},
                timeout=10
            )
            response.raise_for_status()
            return {"provider": "huggingface", "response": response.json()}
        except Exception as e:
            return {"error": str(e)}

class GeminiProvider(LLMProviderBase):
    def __init__(self, provider_name: str, api_keys: List[str]):
        super().__init__(provider_name, api_keys)
        # Configure safety settings to reduce warnings
        self.safety_settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }

    def send_request(
        self,
        prompt: str,
        model: str,
        temperature: float,
        top_p: float,
        **kwargs
    ) -> Dict[str, Any]:
        api_key = self.get_api_key()
        if not api_key:
            return {"error": "No Gemini API key available"}
        
        try:
            # Configure the Gemini API
            genai.configure(api_key=api_key)
            
            # Setup generation config with safety settings
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": 64,
                "max_output_tokens": 8192,
            }
            
            # Create model instance
            model_instance = genai.GenerativeModel(
                model_name=model,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            
            # Start chat session and send message
            chat = model_instance.start_chat(history=[])
            response = chat.send_message(prompt)
            
            return {
                "provider": "gemini",
                "response": {
                    "text": response.text
                }
            }
        except Exception as e:
            return {"error": str(e)}

class LLMManager:
    """
    Main class for handling multiple providers, round-robin key usage, and request logic.
    Supports both library usage and can be integrated with a Flask server.
    """
    def __init__(self, rate_limit: float = 0.0):
        """
        rate_limit (seconds): optional cooldown after each request to avoid throttling.
        """
        self.rate_limit = rate_limit
        self.providers = {}

        # Gather all environment variables as possible keys
        # For example, MISTRAL_KEY1, MISTRAL_KEY2, ...
        self._load_provider_keys("mistral", "MISTRAL_KEY")
        self._load_provider_keys("aistudio", "AISTUDIO_KEY")
        self._load_provider_keys("openrouter", "OPENROUTER_KEY")
        self._load_provider_keys("huggingface", "HF_KEY")
        self._load_provider_keys("gemini", "GEMINI_API_KEY")  # Changed from GEMINI_KEY to GEMINI_API_KEY

    def _load_provider_keys(self, provider_name: str, env_prefix: str):
        keys = []
        index = 1
        while True:
            env_var = f"{env_prefix}{index}"
            if env_var in os.environ:
                keys.append(os.environ[env_var])
                index += 1
            else:
                break

        # If no keys found, check if single key environment variable exists
        if not keys and env_prefix in os.environ:
            keys.append(os.environ[env_prefix])

        if keys:
            if provider_name == "mistral":
                self.providers[provider_name] = MistralProvider(provider_name, keys)
            elif provider_name == "aistudio":
                self.providers[provider_name] = AIStudioProvider(provider_name, keys)
            elif provider_name == "openrouter":
                self.providers[provider_name] = OpenRouterProvider(provider_name, keys)
            elif provider_name == "huggingface":
                self.providers[provider_name] = HuggingFaceProvider(provider_name, keys)
            elif provider_name == "gemini":
                self.providers[provider_name] = GeminiProvider(provider_name, keys)
        else:
            self.providers[provider_name] = None

    def request(
        self,
        prompt: str,
        provider: str,
        model: str,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Primary interface for sending requests to a provider's LLM.
        """
        if provider not in self.providers or not self.providers[provider]:
            return {"error": f"Provider '{provider}' is not configured or has no API keys."}

        time.sleep(self.rate_limit)  # optional cooldown
        handler = self.providers[provider]
        result = handler.send_request(prompt, model, temperature, top_p, **kwargs)
        return result