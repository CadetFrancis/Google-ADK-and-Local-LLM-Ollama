"""Root agent setup and agent team configuration."""
from google.adk.agents import SequentialAgent
from .content_generator_agent import ContentGeneratorAgent
from .tutor_orchestrator_agent import TutorOrchestratorAgent


# Create agent instances
content_generator = ContentGeneratorAgent(
    ollama_endpoint="http://localhost:11434/api/generate",
    ollama_model="llama3.2"
)

tutor_orchestrator = TutorOrchestratorAgent(
    content_generator=content_generator
)

# Create agent team
# The SequentialAgent coordinates the agents, with tutor_orchestrator as the main entry point
# and content_generator available for delegation via transfer_to_agent_tool
# Pass config parameters directly to SequentialAgent
root_agent = SequentialAgent(
    name="language_learning_team",
    description=(
        "A language learning agent team consisting of a tutor orchestrator "
        "that manages the learning loop and a content generator that creates "
        "language learning phrases via Ollama."
    ),
    sub_agents=[tutor_orchestrator, content_generator]
)

# For ADK CLI compatibility, also export as 'agent' if needed
agent = root_agent

