"""Content Generator Agent for generating language learning phrases via Ollama."""
import json
from typing import Optional, Dict, Any
import httpx
from google.adk.agents import LlmAgent


class ContentGeneratorAgent(LlmAgent):
    """
    Agent responsible for generating language learning content using Ollama.
    
    This agent generates phrases and scenario-based content in the target language
    by making HTTP calls to a local Ollama instance.
    """
    
    def __init__(
        self,
        ollama_endpoint: str = "http://localhost:11434/api/generate",
        ollama_model: str = "llama3.2",
        model: str = "gemini-2.5-flash-native-audio-preview-09-2025",
        **kwargs
    ):
        """
        Initialize the ContentGeneratorAgent.
        
        Args:
            ollama_endpoint: The Ollama API endpoint URL
            ollama_model: The Ollama model to use for content generation
            model: The ADK model to use for the agent (default: gemini-2.5-flash-native-audio-preview-09-2025)
            **kwargs: Additional arguments passed to LlmAgent
        """
        # Configure the agent with the specified model
        # Pass config parameters directly to LlmAgent
        super().__init__(
            name="content_generator",
            instruction=(
                "You are a content generator agent that creates language learning phrases "
                "and scenarios using Ollama. Generate appropriate content based on the user's "
                "language, difficulty level, and scenario context."
            ),
            model=model,
            **kwargs
        )
        
        # Use PrivateAttr for non-model fields
        self._ollama_endpoint = ollama_endpoint
        self._ollama_model = ollama_model
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def ollama_endpoint(self) -> str:
        """Get the Ollama endpoint."""
        return self._ollama_endpoint
    
    @property
    def ollama_model(self) -> str:
        """Get the Ollama model."""
        return self._ollama_model
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client for Ollama calls."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def _call_ollama(self, prompt: str) -> str:
        """
        Make an HTTP call to Ollama to generate content.
        
        Args:
            prompt: The prompt to send to Ollama
        
        Returns:
            Generated text from Ollama
        
        Raises:
            RuntimeError: If the Ollama request fails
        """
        client = await self._get_client()
        
        try:
            response = await client.post(
                self.ollama_endpoint,
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            
            result = response.json()
            return result.get("response", "").strip()
        
        except httpx.HTTPError as e:
            raise RuntimeError(f"Failed to call Ollama: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON response from Ollama: {e}") from e
    
    async def generate_phrase(
        self,
        language: str,
        difficulty: str,
        scenario: Optional[str] = None
    ) -> str:
        """
        Generate a single phrase in the target language.
        
        Args:
            language: The target language (e.g., "Spanish", "French")
            difficulty: The difficulty level (e.g., "B1 Intermediate")
            scenario: Optional scenario context (e.g., "ordering food in a restaurant")
        
        Returns:
            Generated phrase text
        
        Raises:
            RuntimeError: If Ollama call fails
        """
        if scenario:
            prompt = (
                f"Generate a single, common {difficulty}-level {language} phrase "
                f"for {scenario}. Return only the phrase, no explanation."
            )
        else:
            prompt = (
                f"Generate a single, common {difficulty}-level {language} phrase. "
                f"Return only the phrase, no explanation."
            )
        
        phrase = await self._call_ollama(prompt)
        return phrase
    
    async def generate_scenario_content(
        self,
        language: str,
        difficulty: str,
        scenario: str
    ) -> Dict[str, Any]:
        """
        Generate multiple phrases for a specific scenario.
        
        Args:
            language: The target language (e.g., "Spanish", "French")
            difficulty: The difficulty level (e.g., "B1 Intermediate")
            scenario: The scenario context (e.g., "checking into a hotel")
        
        Returns:
            Dictionary containing:
            - scenario: The scenario
            - phrases: List of generated phrases
        
        Raises:
            RuntimeError: If Ollama call fails
        """
        prompt = (
            f"Generate 5-7 common {difficulty}-level {language} phrases for {scenario}. "
            f"Return them as a JSON array of strings. Example: [\"phrase1\", \"phrase2\", ...]"
        )
        
        response = await self._call_ollama(prompt)
        
        # Try to parse JSON response
        try:
            # Extract JSON array from response if it's wrapped in text
            import re
            json_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if json_match:
                phrases = json.loads(json_match.group())
            else:
                # Fallback: split by lines or new phrases
                phrases = [line.strip() for line in response.split('\n') if line.strip()]
        except (json.JSONDecodeError, ValueError):
            # Fallback: treat each line as a phrase
            phrases = [line.strip() for line in response.split('\n') if line.strip()]
        
        return {
            "scenario": scenario,
            "phrases": phrases if isinstance(phrases, list) else [phrases]
        }
    
    async def close(self):
        """Close the HTTP client and clean up resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

