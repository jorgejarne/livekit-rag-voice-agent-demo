import logging
import os
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
    inference,
    room_io,
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from llama_index.core import (
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.settings import Settings
from llama_index.embeddings.vertex import VertexTextEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

logger = logging.getLogger("agent")
load_dotenv(".env.local")


def _load_rag_index() -> VectorStoreIndex:
    """Load a RAG index backed by Qdrant + Vertex embeddings.

    Credentials are provided via environment variables (see `env.example`).
    """
    client_email = os.getenv("GOOGLE_VERTEX_EMAIL")
    private_key_id = os.getenv("GOOGLE_VERTEX_PK_ID")
    private_key = os.getenv("GOOGLE_VERTEX_PK")
    model_name_id = os.getenv("GOOGLE_VERTEX_MODEL_ID")
    project_id = os.getenv("GOOGLE_VERTEX_PROJECT_ID")
    location_id = os.getenv("GOOGLE_VERTEX_LOCATION")
    token_endpoint = os.getenv("GOOGLE_VERTEX_TOKEN_ENDPOINT")

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    qdrant_collection = os.getenv("QDRANT_COLLECTION_NAME", "my_docs_vertex")

    missing = [
        name
        for name, val in [
            ("GOOGLE_VERTEX_EMAIL", client_email),
            ("GOOGLE_VERTEX_PK_ID", private_key_id),
            ("GOOGLE_VERTEX_PK", private_key),
            ("GOOGLE_VERTEX_MODEL_ID", model_name_id),
            ("GOOGLE_VERTEX_PROJECT_ID", project_id),
            ("GOOGLE_VERTEX_LOCATION", location_id),
            ("GOOGLE_VERTEX_TOKEN_ENDPOINT", token_endpoint),
            ("QDRANT_URL", qdrant_url),
            ("QDRANT_API_KEY", qdrant_api_key),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(
            "Missing required environment variables for RAG: " + ", ".join(missing)
        )

    embed_model = VertexTextEmbedding(
        model_name=model_name_id,
        project=project_id,
        location=location_id,
        token_uri=token_endpoint,
        client_email=client_email,
        private_key_id=private_key_id,
        private_key=private_key,
    )

    Settings.llm = GoogleGenAI(model="gemini-2.5-flash")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    vector_store = QdrantVectorStore(client=client, collection_name=qdrant_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    return index


class Assistant(Agent):
    def __init__(self, rag_index: VectorStoreIndex) -> None:
        instructions = self.load_prompt(
            Path(__file__).resolve().parent / "assistant_prompt.yaml"
        )
        super().__init__(
            instructions=instructions,
        )
        self.rag_query_engine = rag_index.as_query_engine()

    def load_prompt(self, path: Path) -> str:
        with open(path, encoding="utf-8") as f:
            prompt_yaml = yaml.safe_load(f)

        # For now, we simply stringify the YAML into a structured instruction block
        # This keeps behavior deterministic and easy to debug
        return self.render_prompt(prompt_yaml)

    def render_prompt(self, data: dict) -> str:
        return f"""
        You are {data["agent_identity"]["name"]}.

        Purpose:
        {data["agent_identity"]["purpose"]}

        Interaction Mode:
        {data["interaction_mode"]["type"]}

        Voice Rules:
        {", ".join(data["interaction_mode"]["rules"])}

        Tone:
        {data["persona"]["tone"]}

        Formatting Rules:
        {", ".join(data["formatting_rules"])}

        RAG Rules:
        {", ".join(data["rag_usage_rules"])}

        Priority Rules:
        {", ".join(data["priority_rules"])}

        Booking Trigger Examples:
        {", ".join(data["booking_intent"]["triggers"])}

        Data to Collect:
        {", ".join(data["data_to_collect"])}

        Conversation Order:
        {", ".join(data["conversation_flow"]["order"])}

        Email Rules:
        - Always ask for the email
        - Confirmation required
        - Max spelling attempts: {data["email_validation"]["max_spelling_attempts"]}
        - Must contain @ and .

        Fallback URL:
        {next(iter(data["email_confirmation_flow"]["steps"][-1].values()))["url"]}

        Completion Message:
        {data["completion"]["success_message"]}
        """

    @function_tool
    async def lookup_weather(self, context: RunContext, location: str):
        logger.info(f"Looking up weather for {location}")
        return "sunny with a temperature of 70 degrees."

    @function_tool
    async def ask_company_knowledge(self, ctx: RunContext, question: str) -> str:
        logger.info("RAG retrieval only: %s", question)
        try:
            nodes = self.rag_query_engine.retrieve(question)
            retrieved_chunks = "\n---\n".join(node.text for node in nodes)
            return f"Evidence retrieved from knowledge base:\n{retrieved_chunks}"
        except Exception:
            logger.exception("Error during RAG retrieval")
            return "No relevant knowledge base content found or an error occurred."

    @function_tool
    async def book_consultation_call(
        self, ctx: RunContext, first_name: str, last_name: str, email: str, reason: str
    ) -> str:
        """Book a consultation call: collects first name, last name and email, then sends via Supabase."""
        # Avoid logging PII in public deployments
        logger.info("Booking consultation requested")
        url = os.getenv("SUPABASE_EMAIL_ENDPOINT")
        api_key = os.getenv("SUPABASE_KEY")
        if not url or not api_key:
            logger.error("SUPABASE_EMAIL_ENDPOINT / SUPABASE_KEY not configured")
            return "Booking is not configured on this deployment."

        forced_email_to = os.getenv("BOOKING_FORCE_EMAIL_TO")
        headers = {
            "Content-Type": "application/json",
            "apikey": api_key,
            "Authorization": api_key,
        }

        data = {
            "name": f"{first_name} {last_name}",
            "email": forced_email_to or email,
            "message": reason,
            "subject": "Book consultation",
        }
        try:
            resp = requests.post(url, headers=headers, json=data, timeout=20)
            if resp.status_code == 200 and resp.json().get("success"):
                return "Your consultation request has been received. You and the organizer will receive a confirmation email."
            else:
                logger.error(f"Failed booking: {resp.text}")
                return f"Something went wrong: {resp.text}"
        except Exception:
            logger.exception("Failed to send booking request")
            return "An unexpected error occurred when trying to book your consultation request."


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        # llm=inference.LLM(model="openai/gpt-4.1-mini"),
        llm=inference.LLM(
            model="google/gemini-2.5-flash"
        ),  # exact model string may vary
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    rag_index = _load_rag_index()
    await session.start(
        agent=Assistant(rag_index=rag_index),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()
    await session.say(
        "Hello. I am an AI agent from ProcTrail agency, How can I help you today?",
        allow_interruptions=False,
    )


if __name__ == "__main__":
    cli.run_app(server)
