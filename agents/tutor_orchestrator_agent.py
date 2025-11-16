"""Tutor Orchestrator Agent for managing the language learning loop."""
import base64
from typing import Optional, Dict, Any, Tuple
from google.adk.agents import LlmAgent
from .tools import text_to_speech, speech_to_text, analyze_pronunciation
from .content_generator_agent import ContentGeneratorAgent


class TutorOrchestratorAgent(LlmAgent):
    """
    Main orchestrator agent that manages the language learning loop.
    
    This agent coordinates all aspects of the learning experience:
    - Manages learning session flow
    - Generates and presents phrases to users
    - Converts text to speech (TTS) for pronunciation
    - Analyzes user pronunciation attempts
    - Provides feedback (positive or corrective)
    - Replays phrases on demand
    """
    
    def __init__(
        self,
        content_generator: Optional[ContentGeneratorAgent] = None,
        model: str = "gemini-2.5-flash-native-audio-preview-09-2025",
        **kwargs
    ):
        """
        Initialize the TutorOrchestratorAgent.
        
        Args:
            content_generator: Optional ContentGeneratorAgent instance.
                              If None, will create a default one.
            model: The ADK model to use for the agent (default: gemini-2.5-flash-native-audio-preview-09-2025)
            **kwargs: Additional arguments passed to LlmAgent
        """
        # Configure the agent with the specified model and tools
        # Pass config parameters directly to LlmAgent
        super().__init__(
            name="tutor_orchestrator",
            instruction=(
                "You are a language learning tutor that orchestrates the learning loop. "
                "You manage the learning session, present phrases to users, analyze their "
                "pronunciation, and provide feedback. Use the available tools for text-to-speech, "
                "speech-to-text, and pronunciation analysis."
            ),
            model=model,
            tools=[text_to_speech, speech_to_text, analyze_pronunciation],
            **kwargs
        )
        
        # Use PrivateAttr for non-model fields
        self._content_generator: Optional[ContentGeneratorAgent] = content_generator
        self._user_context: Optional[Dict[str, str]] = None
    
    @property
    def content_generator(self) -> ContentGeneratorAgent:
        """Get or create the content generator agent."""
        if self._content_generator is None:
            self._content_generator = ContentGeneratorAgent()
        return self._content_generator
    
    def set_user_context(
        self,
        language: str,
        difficulty: str,
        scenario: Optional[str] = None
    ) -> None:
        """
        Set the learning context for the session.
        
        Args:
            language: The target language (e.g., "Spanish", "French")
            difficulty: The difficulty level (e.g., "B1 Intermediate")
            scenario: Optional scenario context (e.g., "ordering food in a restaurant")
        """
        self._user_context = {
            "language": language,
            "difficulty": difficulty,
            "scenario": scenario
        }
    
    def _ensure_context(self) -> None:
        """Ensure user context is set before operations that require it."""
        if self._user_context is None:
            raise ValueError(
                "User context not set. Call set_user_context() first."
            )
    
    async def request_new_phrase(self) -> str:
        """
        Request a new phrase from the content generator.
        
        Returns:
            Generated phrase text
        
        Raises:
            ValueError: If user context is not set
            RuntimeError: If phrase generation fails
        """
        self._ensure_context()
        
        phrase = await self.content_generator.generate_phrase(
            language=self._user_context["language"],
            difficulty=self._user_context["difficulty"],
            scenario=self._user_context.get("scenario")
        )
        
        return phrase
    
    async def present_phrase(self, phrase: str) -> bytes:
        """
        Present a phrase with TTS.
        
        Args:
            phrase: The phrase to present
        
        Returns:
            Audio bytes in WAV format
        
        Raises:
            RuntimeError: If TTS fails
        """
        self._ensure_context()
        
        language_code = self._get_language_code(self._user_context["language"])
        # Tools are synchronous, but we can call them in async context
        # Tool returns base64-encoded string, decode to bytes
        audio_base64 = text_to_speech(
            text=phrase,
            language=language_code
        )
        
        return base64.b64decode(audio_base64)
    
    async def analyze_user_speech(
        self,
        user_audio: bytes,
        target_phrase: str
    ) -> Dict[str, Any]:
        """
        Analyze user's pronunciation.
        
        Args:
            user_audio: User's audio recording as bytes
            target_phrase: The target phrase to compare against
        
        Returns:
            Analysis dictionary with accuracy, is_correct, feedback, etc.
        
        Raises:
            RuntimeError: If analysis fails
        """
        self._ensure_context()
        
        language_code = self._get_language_code(self._user_context["language"])
        
        # Encode audio bytes to base64 string for tool
        audio_base64 = base64.b64encode(user_audio).decode('utf-8')
        
        # First, convert speech to text
        user_text = speech_to_text(
            audio=audio_base64,
            language=language_code
        )
        
        # Then, analyze pronunciation
        analysis = analyze_pronunciation(
            user_audio=audio_base64,
            target_text=target_phrase,
            language=language_code
        )
        
        return analysis
    
    async def provide_feedback(
        self,
        analysis: Dict[str, Any],
        target_phrase: str
    ) -> Tuple[str, bytes]:
        """
        Generate feedback based on analysis.
        
        Args:
            analysis: Pronunciation analysis results
            target_phrase: The target phrase
        
        Returns:
            Tuple of (feedback_text, feedback_audio)
        
        Raises:
            RuntimeError: If feedback generation fails
        """
        self._ensure_context()
        
        if analysis.get("is_correct", False):
            feedback_text = f"¡Perfecto! Great job! You pronounced '{target_phrase}' correctly."
        else:
            problematic_words = analysis.get("problematic_words", [])
            suggestions = analysis.get("suggestions", [])
            
            if problematic_words:
                words_str = ", ".join(problematic_words)
                feedback_text = (
                    f"That was close! Let's focus on the word(s): {words_str}. "
                )
            else:
                feedback_text = "That was close! "
            
            if suggestions:
                feedback_text += " ".join(suggestions)
            else:
                feedback_text += "Try to match the pronunciation more closely."
        
        language_code = self._get_language_code(self._user_context["language"])
        # Tool returns base64-encoded string, decode to bytes
        feedback_audio_base64 = text_to_speech(
            text=feedback_text,
            language=language_code
        )
        feedback_audio = base64.b64decode(feedback_audio_base64)
        
        return feedback_text, feedback_audio
    
    async def replay_phrase(self, phrase: str) -> bytes:
        """
        Replay a phrase.
        
        Args:
            phrase: The phrase to replay
        
        Returns:
            Audio bytes in WAV format
        
        Raises:
            RuntimeError: If TTS fails
        """
        return await self.present_phrase(phrase)
    
    def _get_language_code(self, language: str) -> str:
        """
        Convert language name to language code.
        
        Args:
            language: Language name (e.g., "Spanish", "French")
        
        Returns:
            Language code (e.g., "es", "fr")
        """
        language_map = {
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "English": "en",
            "Chinese": "zh",
            "Japanese": "ja",
            "Korean": "ko",
            "Russian": "ru"
        }
        return language_map.get(language, "en")
    
    async def close(self):
        """Close resources and clean up."""
        if self._content_generator is not None:
            await self._content_generator.close()

