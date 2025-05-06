from dotenv import load_dotenv
import asyncio
import logging
import os
import requests
import google.generativeai as genai
from gtts import gTTS
import tempfile
from livekit import rtc
from livekit import agents
from livekit.agents import AgentSession, Agent, AutoSubscribe, RunContext, RoomInputOptions, llm, stt as livekit_stt, tts as livekit_tts, vad as livekit_vad
from livekit.plugins import (
    cartesia,
    deepgram,
    silero,groq,
    noise_cancellation
)
from livekit.agents.llm import ChatContext, ChatMessage, StopResponse
# from livekit.agents import pipeline
from livekit.agents.llm import function_tool
# from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv()

logger = logging.getLogger("transcriber")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") #Replace with your key.
genai.configure(api_key=GOOGLE_API_KEY)
# models = genai.list_models()

# for model in models:
#     if 'generateContent' in model.supported_generation_methods:
#         print(model.name)

model = genai.GenerativeModel('gemini-2.0-flash')

def prewarm(proc: agents.JobProcess):
    print("PREWARM FUNCTION CALLED!")
    # preload models when process starts to speed up first interaction
    proc.userdata["vad"] = silero.VAD.load()

    # fetch cartesia voices

    headers = {
        "X-API-Key": os.getenv("CARTESIA_API_KEY", ""),
        "Cartesia-Version": "2024-08-01",
        "Content-Type": "application/json",
    }
    response = requests.get("https://api.cartesia.ai/voices", headers=headers)
    if response.status_code == 200:
        print("Connected")
        proc.userdata["cartesia_voices"] = response.json()
    else:
        print(f"Error: Failed to fetch Cartesia voices: {response.status_code}")
        logger.warning(f"Failed to fetch Cartesia voices: {response.status_code}")


class Assistant(Agent):
    def __init__(self, session: AgentSession, room: rtc.Room, stt_engine: livekit_stt.STT, llm_engine: llm.LLM, tts_engine: livekit_tts.TTS, vad_engine: livekit_vad.VAD, turn_detector: MultilingualModel) -> None:
        super().__init__(
            instructions="You are a helpful voice AI assistant.Note: If asked to print to the console, use the `print_to_console` function.",
            stt=stt_engine,
            llm=llm_engine,
            tts=tts_engine,
            vad=vad_engine,
            turn_detection=turn_detector
        )
        print("Assistant Init")
        # self.session = session  # Keep a reference to the session
        # self.room = room
    # def __init__(self) -> None:
    #     super().__init__(instructions="You are a helpful voice AI assistant.")
    #     print("Assistant initialized")
    async def on_enter(self):
        print("Enter")
        await self.session.say("How are you?")
        # await self.session.generate_reply(user_input="Say something somewhat long and boring so I can test")

    async def on_user_turn_completed(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        # callback before generating a reply after user turn committed
        print("on_user_turn_completed")
        await self.session.say("Hello")
        if not new_message.text_content:
            # for example, raise StopResponse to stop the agent from generating a reply
            logger.info("ignore empty user turn")
            raise StopResponse()
    @function_tool
    async def print_to_console(self, context: RunContext):
        print("Console Print Success!")
        return None, "I've printed to the console."
    async def on_end_of_turn(self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage, generating_reply: bool) -> None:
        logger.info(f"Assistant received user message: {new_message.content}")
        print("EndofTurn")
        try:
            response = model.generate_content(new_message.content)
            ai_response = response.text
            logger.info(f"Assistant LLM Response: {ai_response}")

            # Generate TTS audio and say it
            await self.session.say(ai_response)

        except Exception as e:
            logger.error(f"Assistant error processing turn: {e}")

    async def on_exit(self) -> None:
        print("Exit")
    async def publish_audio(self, room: rtc.Room, audio_data: bytes):
        # This function might not be needed directly anymore if 'session.say' handles publishing.
        logger.info("Assistant publishing generated audio...")
        try:
            audio_track = rtc.LocalAudioTrack.create_opus_track(name="agent_audio")
            publication = await room.local_participant.publish_track(audio_track)
            await audio_track.sink.write_frame(audio_data) # You might need to adjust this based on audio format
            duration_seconds = len(audio_data) / (48000 * 2 * 2) if audio_data else 1
            await asyncio.sleep(duration_seconds + 0.5) # Add a small buffer
            await room.local_participant.unpublish_track(publication.sid)
            logger.info("Assistant audio published successfully.")
        except Exception as e:
            logger.error(f"Assistant error publishing audio: {e}")


async def entrypoint(ctx: agents.JobContext):
    logger.info(f"starting transcriber (STT), room: {ctx.room.name}")

    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    participant = await ctx.wait_for_participant()
    print("Participant joined: id",participant.identity)
    logger.info(f"starting voice assistant for participant {participant.identity}")

    stt_engine = deepgram.STT(model="nova-2-general", language="en-US", interim_results=True, smart_format=True, punctuate=True, filler_words=True, profanity_filter=False, keywords=[("LiveKit", 1.5)])
    tts_engine = cartesia.TTS(model="sonic-2")
    vad_engine = silero.VAD.load()
    turn_detector = MultilingualModel()

    # assistant = Assistant(None, ctx.room, stt_engine, model, tts_engine, vad_engine, turn_detector) # Pass engines to Assistant

    is_user_speaking = False
    is_agent_speaking = False

    session = AgentSession(
        turn_detection=turn_detector,
        stt=stt_engine,
        vad=vad_engine,
        llm=groq.LLM(model="llama-3.3-70b-versatile"),#model,
        tts=tts_engine,
    )
    # assistant.session = session # Now it might be okay to set the session here

    agent1 = Agent(
        instructions="""
            You are a friendly voice assistant built by LiveKit.
            Start every conversation by greeting the user.
            Only use the `lookup_weather` tool if the user specifically asks for weather information.
            Never assume a location or provide weather data without a request.
            """,)
    agent = Assistant(session=session, room=ctx.room, stt_engine=stt_engine, llm_engine=model, tts_engine=tts_engine, vad_engine=vad_engine, turn_detector=turn_detector)
    await session.start(
        room=ctx.room,
        agent=agent
        # room_input_options=RoomInputOptions(
        #     noise_cancellation=None, # Noise cancellation removed
        # ),
    )
    # asyncio.create_task(
    #     session.start(
    #         agent=Assistant(),
    #         room=ctx.room,
    #         room_input_options=RoomInputOptions(
    #             # enable Krisp background voice and noise removal
    #             noise_cancellation=noise_cancellation.BVC(),
    #         ),
    #     )
    # )
    # await session.say("Hey, how can I help you today?", allow_interruptions=True)
    # await session.generate_reply(
    #     instructions="Greet the user and offer your assistance."
    # )
   
    @ctx.room.on("data_received")
    def on_data_received(packet):
        print("data_Received")

    @ctx.room.on("track_subscribed")
    def on_data_received(packet):
        print("track_subscribed")

if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm) )