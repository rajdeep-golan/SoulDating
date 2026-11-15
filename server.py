from dotenv import load_dotenv
import asyncio
import logging
import os

from livekit import agents
from livekit.agents import AgentSession, Agent, RunContext, WorkerOptions, cli, function_tool, llm, stt as livekit_stt, tts as livekit_tts, vad as livekit_vad
from livekit.plugins import (
    deepgram,
    silero,
    groq,
    elevenlabs,
)
# from livekit.plugins.turn_detector.multilingual import MultilingualModel

from message_logger import log_message
from biodataExtrator import extract_biodata
from profile_capture import save_call_profile, transcript_from_history

load_dotenv()
logger = logging.getLogger("soul_agent")


class SoulInfoAgent(Agent):
    def __init__(self, session: AgentSession, stt_engine, llm_engine, tts_engine, vad_engine, turn_detector) -> None:
        super().__init__(
            instructions="""
                You have a name: Zoey. You are really smart in terms of Love and Connections. You have almost 100% success in connecting perfect couple.
                Right now, Start with giving your introduction and telling that you are on a mission to promote Trust, Loyalty and Respect and help you find the best Soul Match possible. Your purpose is to collect information from the user. Sometimes use fillers "uhm" and "ahh" to make it sound more natural and try to be funny and joyful.
                Specifically, you should ask for the following information:

                1. Name
                2. Age
                3. Gender, and who they would like to be matched with
                4. Hometown (and where they live now)
                5. Likes / hobbies
                6. Dream City
                7. Dislikes
                8. Height
                9. What they are looking for in a relationship (long-term, marriage, casual)

                Start by saying - Let's create your profile. Engage in a natural conversation. Do not ask all the questions at once. Ask one question, and wait for the user's response. Acknowledge the user's response, and then ask the next question.
                Try to vary your questions, and make them sound natural. For example, do not always say "What is your...".
                Keep every reply short (one or two sentences). Once you have collected all the information, say "Thank you, I have collected all the information. I am finding your best soul matches now!" and end the conversation.
            """,
            stt=stt_engine,
            llm=llm_engine,
            tts=tts_engine,
            vad=vad_engine,
            turn_detection=turn_detector,
        )
        self.collected_info = {}
        self.questions = [
            "What is your name?",
            "Where is your hometown?",
            "What are some things you like?",
            "What city would you love to live in someday?",
            "What are some things you dislike?",
            "How tall are you?"
        ]
        self.current_question_index = 0
        self.user_id = None  # Set dynamically during conversation

    async def on_enter(self):
        await self.session.say("Hello! I'm Zoey, I am on a mission to promote Trust, Loyalty and Respect and help you find the best Soul Match possible. Let's get to know you a little better. Tell me something cool about you.")
        await self.session.generate_reply(instructions=self.questions[self.current_question_index])

    async def on_end_of_turn(self, chat_ctx, new_message, generating_reply: bool):
        try:
            # Log user message
            if new_message.role == "user":
                self.user_id = chat_ctx.session_id or "default_user"
                log_message(self.user_id, new_message.role, new_message.content)

            if self.current_question_index < len(self.questions):
                question_key = self.questions[self.current_question_index].lower()\
                    .replace("what is your ", "")\
                    .replace("what are some things you ", "")\
                    .replace("?", "")\
                    .replace("what city would you love to live in someday", "dream city")
                
                self.collected_info[question_key] = new_message.content
                await self.session.say(f"Okay, I have that your {question_key} is {new_message.content}.")
                log_message(self.user_id, "assistant", f"Okay, I have that your {question_key} is {new_message.content}.")

                self.current_question_index += 1

                if self.current_question_index < len(self.questions):
                    await self.session.generate_reply(instructions=self.questions[self.current_question_index])
                else:
                    await self.session.say("Thank you, I have collected all the information.")
                    log_message(self.user_id, "assistant", "Thank you, I have collected all the information.")
                    
                    # 🧠 Extract biodata using LLM
                    biodata = extract_biodata(self.user_id)
                    logger.info("Extracted biodata: %s", biodata)

        except Exception as e:
            logger.error(f"Error in on_end_of_turn: {e}")

    async def on_exit(self):
        print("Conversation ended. Collected info:", self.collected_info)


async def entrypoint(ctx: agents.JobContext):
    logger.info(f"Starting Soul Info Agent, room: {ctx.room.name}")
    await ctx.connect()

    stt_engine = deepgram.STT(
        model="nova-2-general",
        language="en-US",
        interim_results=True,
        smart_format=True,
        punctuate=True,
        filler_words=True,
        profanity_filter=False,
    )

    if os.getenv("ELEVEN_API_KEY", "").startswith("sk_"):
        tts_engine = elevenlabs.TTS(
            voice_id="Zjz30d9v1e5xCxNVTni6",
            model="eleven_multilingual_v2"
        )
    else:
        # Fallback voice: Deepgram Aura (uses DEEPGRAM_API_KEY)
        tts_engine = deepgram.TTS(model="aura-2-thalia-en")

    llm_engine = groq.LLM(model="openai/gpt-oss-120b")
    vad_engine = silero.VAD.load()
    turn_detector = "vad" #MultilingualModel()

    session = AgentSession(
        stt=stt_engine,
        tts=tts_engine,
        llm=llm_engine,
        vad=vad_engine,
        turn_detection=turn_detector,
    )

    agent = SoulInfoAgent(
        session=session,
        stt_engine=stt_engine,
        llm_engine=llm_engine,
        tts_engine=tts_engine,
        vad_engine=vad_engine,
        turn_detector=turn_detector,
    )

    captured = {"done": False}

    async def _capture_profile(*_):
        if captured["done"]:
            return
        captured["done"] = True
        transcript = transcript_from_history(session.history)
        logger.info("capturing profile from %d transcript lines", len(transcript))
        await asyncio.to_thread(save_call_profile, transcript, ctx.room.name)

    async def _on_close():
        await _capture_profile()
        ctx.shutdown(reason="call ended")

    # Hang-up closes the session -> extract profile -> store in DB
    session.on("close", lambda _ev: asyncio.create_task(_on_close()))
    ctx.add_shutdown_callback(_capture_profile)

    await session.start(agent=agent, room=ctx.room)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, agent_name=os.getenv("AGENT_NAME", "")))
