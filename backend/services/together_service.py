import os
import asyncio
import concurrent.futures
import time
from typing import List, Optional, Dict, Any
import together
from utils.logger import get_logger
from utils.logging_config import log_performance
from config.settings import settings

# Thread pool for concurrent requests
together_pool = concurrent.futures.ThreadPoolExecutor(max_workers=20)

# Get logger for this service
together_logger = get_logger("together_service")


class TogetherService:
    """Service for interacting with Together.ai API"""

    @staticmethod
    def get_api_key() -> str:
        """Get Together.ai API key from settings"""
        return settings.TOGETHER_API_KEY

    @staticmethod
    def get_model() -> str:
        """Get Together.ai model from settings"""
        return settings.TOGETHER_MODEL

    @staticmethod
    def get_base_url() -> str:
        """Get Together.ai base URL from settings"""
        return settings.TOGETHER_BASE_URL

    @staticmethod
    def initialize_client() -> together.Together:
        """Initialize and return Together.ai client"""
        api_key = TogetherService.get_api_key()
        base_url = TogetherService.get_base_url()

        together_logger.debug(
            "Initializing Together.ai client",
            extra={"extra_fields": {"api_key_set": bool(api_key), "base_url": base_url}}
        )
        if not api_key:
            together_logger.error("TOGETHER_API_KEY is not set in settings")
            raise ValueError("TOGETHER_API_KEY environment variable is required")

        client = together.Together(api_key=api_key, base_url=base_url)
        together_logger.debug("Together.ai client initialized successfully")
        return client

    @staticmethod
    async def generate_completion(
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> str:
        """
        Generate completion using Together.ai API

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use (defaults to settings)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            **kwargs: Additional arguments for the API

        Returns:
            Generated text response
        """
        loop = asyncio.get_event_loop()
        api_key = TogetherService.get_api_key()
        model = model or TogetherService.get_model()

        if not api_key:
            raise ValueError("Together.ai API key not configured")

        def _generate():
            try:
                client = TogetherService.initialize_client()

                # Prepare the request parameters
                request_params = {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                }

                if max_tokens:
                    request_params["max_tokens"] = max_tokens

                # Add any additional kwargs
                request_params.update(kwargs)

                together_logger.debug(
                    "Sending completion request to Together.ai",
                    extra={"extra_fields": {
                        "model": model,
                        "message_count": len(messages),
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    }}
                )

                response = client.chat.completions.create(**request_params)
                result = response.choices[0].message.content

                together_logger.debug(
                    "Successfully received completion from Together.ai",
                    extra={"extra_fields": {
                        "response_length": len(result) if result else 0,
                        "model_used": model
                    }}
                )

                return result, None

            except Exception as e:
                together_logger.error(
                    "Together.ai API error",
                    extra={"extra_fields": {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "model": model
                    }}
                )
                return None, e

        try:
            result, error = await loop.run_in_executor(together_pool, _generate)

            if error:
                raise error

            if not result:
                raise ValueError("No response generated from Together.ai")

            return result

        except Exception as e:
            together_logger.error(
                "Failed to generate completion",
                extra={"extra_fields": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "model": model
                }}
            )
            raise

    @staticmethod
    async def generate_chat_response(
        user_message: str,
        system_message: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        **kwargs,
    ) -> str:
        """
        Generate a chat response using Together.ai

        Args:
            user_message: The user's message
            system_message: Optional system message for context
            model: Model to use (defaults to settings)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments

        Returns:
            AI response text
        """
        messages = []

        if system_message:
            messages.append({"role": "system", "content": system_message})

        messages.append({"role": "user", "content": user_message})

        return await TogetherService.generate_completion(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

    @staticmethod
    async def check_api_health() -> bool:
        """
        Check if Together.ai API is accessible and working

        Returns:
            True if API is healthy, False otherwise
        """
        try:
            api_key = TogetherService.get_api_key()
            if not api_key:
                return False

            loop = asyncio.get_event_loop()

            def _health_check():
                try:
                    client = TogetherService.initialize_client()
                    # Simple API call to check health
                    models = client.models.list()
                    return len(models) > 0, None
                except Exception as e:
                    return False, e

            result, error = await loop.run_in_executor(together_pool, _health_check)

            if error:
                together_logger.error(f"Together.ai health check failed: {str(error)}")
                return False

            return result

        except Exception as e:
            together_logger.error(f"Health check error: {str(e)}")
            return False

    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of available models from Together.ai

        Returns:
            List of model names
        """
        try:
            client = TogetherService.initialize_client()
            models = client.models.list()
            return [model.id for model in models if hasattr(model, "id")]
        except Exception as e:
            together_logger.error(f"Failed to get models: {str(e)}")
            return []
