from dotenv import load_dotenv
import asyncio
import logging
import os
import requests  # Import requests for fetching data (if needed)
from livekit import rtc
from livekit import agents
from livekit.agents import AgentSession, Agent, RunContext, WorkerOptions, cli, function_tool, llm, stt as livekit_stt, tts as livekit_tts, vad as livekit_vad
from livekit.plugins import (
    cartesia,
    deepgram,
    silero,groq,
    noise_cancellation
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel  # Import turn detector

load_dotenv()

logger = logging.getLogger("soul_agent")
# lk app create \
# 	--sandbox 
# lk app create --template voice-assistant-swift --sandbox interactive-blockchain-1muog9
def prewarm(proc: agents.JobProcess):
    print("PREWARM FUNCTION CALLED!")
    proc.userdata["vad"] = silero.VAD.load()
    #  Consider pre-loading LLM if needed for faster initial response
    #  proc.userdata["llm"] = groq.LLM(model="llama-3.3-70b-versatile")

    # fetch cartesia voices (optional, if you need this in prewarm)
    try:
        headers = {
            "X-API-Key": os.getenv("CARTESIA_API_KEY", ""),
            "Cartesia-Version": "2024-08-01",
            "Content-Type": "application/json",
        }
        response = requests.get("https://api.cartesia.ai/voices", headers=headers)
        if response.status_code == 200:
            print("Connected to Cartesia API")
            proc.userdata["cartesia_voices"] = response.json()
        else:
            print(f"Error: Failed to fetch Cartesia voices: {response.status_code}")
            logger.warning(f"Failed to fetch Cartesia voices: {response.status_code}")
    except Exception as e:
        logger.error(f"Error during prewarm Cartesia fetch: {e}")



class SoulInfoAgent(Agent):
    def __init__(self, session: AgentSession, stt_engine: livekit_stt.STT, llm_engine: llm.LLM, tts_engine: livekit_tts.TTS, vad_engine: livekit_vad.VAD, turn_detector: MultilingualModel) -> None:
        super().__init__(
            instructions="""
                You have a name: Soul. You are really smart in terms of Love and Connections. You have almost 100% success in connecting perfect couple.
                Right now, Your purpose is to collect information from the user.
                Specifically, you should ask for the following information:

                1.  Name
                2.  Hometown
                3.  Likes
                4.  Dream City
                5.  Dislikes
                6.  Height

                Start by saying - Let's create your profile . Engage in a natural conversation.  Do not ask all the questions at once. Ask one question, and wait for the user's response.  Acknowledge the user's response, and then ask the next question.
                Try to vary your questions, and make them sound natural.  For example, do not always say "What is your...".
                Once you have collected all the information, say "Thank you, I have collected all the information." and end the conversation.
                """,
            stt=stt_engine,
            llm=llm_engine,
            tts=tts_engine,
            vad=vad_engine,
            turn_detection=turn_detector,  # Pass turn detector
        )
        # self.session = session
        self.collected_info = {}
        self.questions = [
            "What is your name?",
            "What city would you love to live in someday?",  # More natural
            "Where is your hometown?",  # More natural
            "What are some things you like?",
            "What are some things you dislike?",
            "How tall are you?",  # More natural
        ]
        self.current_question_index = 0

    async def on_enter(self):
        print("Enter")
        await self.session.say("Hello! I'm Soul, and I'd like to get to know you a little better. Tell me something cool about you.")
        await self.session.generate_reply(instructions=self.questions[self.current_question_index])

    async def on_end_of_turn(self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage, generating_reply: bool) -> None:
        print("End of Turn - Called!")  # Debug: Confirm this is called
        print(f"New Message: {new_message}")  # Debug: Inspect the message
        logger.info(f"Agent received user message: {new_message.content}")
        try:
            # Process user response
            if self.current_question_index < len(self.questions):
                question_key = self.questions[self.current_question_index].lower().replace("what is your ", "").replace("what are some things you ", "").replace("?", "")
                self.collected_info[question_key] = new_message.content
                await self.session.say(f"Okay, I have that you {question_key} is {new_message.content}.") # Acknowledge

                self.current_question_index += 1
                if self.current_question_index < len(self.questions):
                    await self.session.generate_reply(instructions=self.questions[self.current_question_index])
                else:
                    await self.session.say("Thank you, I have collected all the information.")

        except Exception as e:
            logger.error(f"Error in on_end_of_turn: {e}")

    async def on_exit(self) -> None:
        print("Exit")
        print(f"Collected Information: {self.collected_info}") # Print Collected Info.

async def entrypoint(ctx: agents.JobContext):
    print("Entry Point!")
    logger.info(f"starting Soul Info Agent, room: {ctx.room.name}")
    await ctx.connect()

    stt_engine = deepgram.STT(model="nova-2-general", language="en-US", interim_results=True, smart_format=True, punctuate=True, filler_words=True, profanity_filter=False, keywords=[("LiveKit", 1.5)])
    tts_engine = cartesia.TTS(model="sonic-2")
    vad_engine = silero.VAD.load()
    turn_detector = MultilingualModel()  # Initialize turn detector

    session = AgentSession(
        turn_detection=turn_detector,  # Pass turn detector to session
        stt=stt_engine,
        vad=vad_engine,
        llm=groq.LLM(model="llama-3.3-70b-versatile"),  # Or your preferred LLM
        tts=tts_engine,
    )

    agent = SoulInfoAgent(session=session, stt_engine=stt_engine, llm_engine=groq.LLM(model="llama-3.3-70b-versatile"), tts_engine=tts_engine, vad_engine=vad_engine, turn_detector=turn_detector)
    await session.start(agent=agent, room=ctx.room)



if __name__ == "__main__":
    print("PSSSSREWARM FUNCTION CALLED!")
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
