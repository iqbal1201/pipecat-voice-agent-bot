#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Bot Implementation.

This module implements a chatbot using Google's Gemini Multimodal Live model.
It includes:
- Real-time audio/video interaction through Daily
- Animated robot avatar
- Speech-to-speech model

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow using Gemini's streaming capabilities.
"""

import asyncio
import os
import sys
import requests
import json

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    OutputImageRawFrame,
    SpriteFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.google.stt import GoogleSTTService
from pipecat.services.google.tts import GoogleTTSService
from pipecat.services.google.llm import GoogleLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.transcriptions.language import Language

load_dotenv(override=True)

FLASK_BACKEND_URL = os.getenv("FLASK_BACKEND_URL")
_agent_deployment_id: str = None

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

sprites = []
script_dir = os.path.dirname(__file__)

for i in range(1, 26):
    # Build the full path to the image file
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    # Get the filename without the extension to use as the dictionary key
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

# Create a smooth animation by adding reversed frames
flipped = sprites[::-1]
sprites.extend(flipped)

# Define static and animated states
quiet_frame = sprites[0]  # Static frame for when bot is listening
talking_frame = SpriteFrame(images=sprites)  # Animation sequence for when bot is talking


class TalkingAnimation(FrameProcessor):
    """Manages the bot's visual animation states.

    Switches between static (listening) and animated (talking) states based on
    the bot's current speaking status.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and update animation state.

        Args:
            frame: The incoming frame to process
            direction: The direction of frame flow in the pipeline
        """
        await super().process_frame(frame, direction)

        # Switch to talking animation when bot starts speaking
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        # Return to static frame when bot stops speaking
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame, direction)


def set_deployment_id(deployment_id: str):
    global _agent_deployment_id
    _agent_deployment_id = deployment_id


def fetch_agent_config(deployment_id: str) -> dict:
    url = f"{FLASK_BACKEND_URL}/api/get-agent-data/{deployment_id}"
    logger.info(f"Fetching agent config from {url}")
    # response = requests.get(url)
    # response.raise_for_status()
    # return response.json()
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
        agent_data = response.json()

        # --- Map Language string from Flask to Pipecat Language Enum and Google Language Code ---
        lang_map = {
            "English": {"enum": Language.EN_US, "code": "en-US"},
            "Bahasa Indonesia": {"enum": Language.ID, "code": "id-ID"},
            "Hindi": {"enum": Language.HI, "code": "hi-IN"},
            # Add more language mappings as needed for your agent
        }

        # --- Define dynamic initial greeting templates (with agent name placeholder) ---
        initial_greetings_templates = {
            Language.EN_US: "Hello! I am {agent_name}, your AI assistant. How can I help you today?",
            Language.ID: "Halo! Saya {agent_name}, asisten AI Anda. Ada yang bisa saya bantu hari ini?",
            Language.HI: "नमस्ते! मैं {agent_name}, आपका एआई असिस्टेंट हूँ। मैं आज आपकी क्या मदद कर सकता हूँ?",
        }

        # --- Define LLM System Instruction Templates (with placeholders for dynamic content) ---
        llm_instruction_templates = {
            Language.EN_US: {
                "intro": "You are an AI assistant named {agent_name}. Your gender is {gender}. Your primary role is to {purpose}. Your business context is: {business_context}. You possess traits: {traits}. Your flaws include: {flaws}.",
                "language_directive": "Always use English in all your responses. Do not use any other language or mixed languages. Start every response directly in English.",
                "general_directives": "Respond to what the user said in a creative and helpful way. Do not include special characters like (*) unless required such as (@ in an email address). Prioritize answering common questions from your knowledge base. Your knowledge base is: {knowledge_base_content}.",
                "campaign_directives": "Additional instructions for conversation flow and function calling: {campaign_design_prompt}.",
                "persona_content_placeholder": "{ai_generated_content}"  # This will be the AI-generated part from Flask
            },
            Language.ID: {
                "intro": "Anda adalah asisten AI bernama {agent_name}. Gender Anda adalah {gender}. Peran utama Anda adalah {purpose}. Konteks bisnis Anda adalah: {business_context}. Anda memiliki sifat-sifat: {traits}. Kekurangan Anda adalah: {flaws}.",
                "language_directive": "Selalu gunakan Bahasa Indonesia dalam semua respons Anda. Jangan gunakan bahasa lain atau campuran bahasa. Mulai setiap respons langsung dalam Bahasa Indonesia.",
                "general_directives": "Tanggapi apa yang dikatakan pengguna dengan cara yang kreatif dan membantu. Jangan sertakan karakter khusus seperti (*) kecuali diperlukan seperti (@ di alamat email). Prioritaskan menjawab pertanyaan umum dari basis pengetahuan Anda. Berikut adalah basis pengetahuan Anda: {knowledge_base_content}.",
                "campaign_directives": "Instruksi tambahan untuk alur percakapan dan pemanggilan fungsi: {campaign_design_prompt}.",
                "persona_content_placeholder": "{ai_generated_content}"
            },
            Language.HI: {
                "intro": "आप {agent_name} नामक एक AI सहायक हैं। आपका लिंग {gender} है। आपकी प्राथमिक भूमिका है {purpose}। आपका व्यावसायिक संदर्भ है: {business_context}। आप में ये विशेषताएँ हैं: {traits}। आपकी कमियाँ शामिल हैं: {flaws}।",
                "language_directive": "हमेशा अपनी सभी प्रतिक्रियाओं में हिंदी का उपयोग करें। किसी अन्य भाषा या मिश्रित भाषाओं का उपयोग न करें। अपनी प्रत्येक प्रतिक्रिया सीधे हिंदी में शुरू करें।",
                "general_directives": "उपयोगकर्ता ने जो कहा, उसका रचनात्मक और सहायक तरीके से जवाब दें। विशेष वर्णों जैसे (*) का उपयोग न करें जब तक कि आवश्यक न हो जैसे (@ ईमेल पते में)। अपने ज्ञान आधार से सामान्य प्रश्नों का उत्तर देने को प्राथमिकता दें। आपका ज्ञान आधार है: {knowledge_base_content}।",
                "campaign_directives": "संवाद प्रवाह और फ़ंक्शन कॉलिंग के लिए अतिरिक्त निर्देश: {campaign_design_prompt}।",
                "persona_content_placeholder": "{ai_generated_content}"
            }
        }

        # --- Validate and retrieve language information ---
        selected_language_name = agent_data['parameters']['language']
        lang_info = lang_map.get(selected_language_name)

        if not lang_info:
            raise ValueError(f"Unsupported language '{selected_language_name}' found in agent config. "
                             "Please update lang_map in fetch_agent_config.")

        # --- Extract agent details ---
        agent_name = agent_data['parameters']['name']
        agent_gender_raw = agent_data['parameters'].get('gender', 'Female')
        # Ensure gender is in uppercase as expected by GoogleTTS.InputParams.gender
        agent_gender_param = agent_gender_raw.upper()
        if agent_gender_param not in ["MALE", "FEMALE", "NEUTRAL"]:
            agent_gender_param = "FEMALE"  # Default to FEMALE if invalid or not provided

        # --- Format initial greeting ---
        greeting_template = initial_greetings_templates.get(lang_info["enum"],
                                                            initial_greetings_templates[Language.EN_US])
        initial_greeting = greeting_template.format(agent_name=agent_name)

        # --- Map Voice Speed and Age to Speaking Rate (for Google TTS AudioConfig) ---
        speaking_rate_map = {
            "Normal": 1.0,
            "Fast": 1.2,
            "Slow": 0.8,
            "Youthful": 1.1,  # Using Voice Age to influence initial speed, if not explicitly "Fast/Slow"
            "Mature": 0.95,
            "Elderly": 0.85,
        }
        voice_speed_text = agent_data['parameters'].get('voiceSpeed', 'Normal')
        speaking_rate_val = speaking_rate_map.get(voice_speed_text, 1.0)
        voice_age_text = agent_data['parameters'].get('voiceAge', '')
        # If voice speed wasn't explicitly set (i.e., it's 'Normal'), and voice age has a speed suggestion, use it.
        if voice_speed_text == 'Normal' and voice_age_text in speaking_rate_map:
            speaking_rate_val = speaking_rate_map[voice_age_text]

        # --- Construct the DYNAMIC LLM System Instruction ---
        # Get the instruction templates for the selected language
        selected_llm_instructions = llm_instruction_templates.get(lang_info["enum"],
                                                                  llm_instruction_templates[Language.EN_US])

        llm_system_instruction = (
                selected_llm_instructions["intro"].format(
                    agent_name=agent_name,
                    gender=agent_gender_raw.lower(),  # Use lowercase for natural language within prompt
                    purpose=agent_data['parameters']['purpose'],
                    business_context=agent_data['parameters']['aboutGame'],
                    traits=agent_data['parameters']['traits'],
                    flaws=agent_data['parameters']['flaws']
                ) + " " +
                selected_llm_instructions["language_directive"] + " " +
                selected_llm_instructions["general_directives"].format(
                    knowledge_base_content=agent_data['parameters']['knowledgeBaseContent']
                ) + " " +
                selected_llm_instructions["campaign_directives"].format(
                    campaign_design_prompt=agent_data['parameters']['campaignDesignPrompt']
                ) + " " +
                selected_llm_instructions["persona_content_placeholder"].format(
                    ai_generated_content=agent_data['ai_generated_content']
                )
        ).strip()  # .strip() removes any leading/trailing whitespace

        # --- Construct the pipecat_config dictionary with all derived parameters ---
        pipecat_config = {
            "name": agent_name,  # 1. Name of Agent
            "gender": agent_gender_param,  # 2. Gender (uppercase for TTS)
            "use_case": agent_data['parameters']['useCase'],  # 3. Use Case
            "language_enum": lang_info["enum"],  # 4. Language (Pipecat Enum)
            "language_code": lang_info["code"],  # 4. Language (Google Code)
            "purpose": agent_data['parameters']['purpose'],  # 5. Purpose of the Agent
            "business_context": agent_data['parameters']['aboutGame'],  # 6. Business Context
            "traits": agent_data['parameters']['traits'],  # 7. Traits of the Agent
            "flaws": agent_data['parameters']['flaws'],  # 8. Flaws
            "knowledge_base_content": agent_data['parameters']['knowledgeBaseContent'],  # 9. Knowledge Base
            "voice_id": agent_data['parameters']['voiceSelect'],  # 10. Voice type
            "speaking_rate": speaking_rate_val,  # 11. Voice Speed
            "llm_model": agent_data['parameters']['llm'],  # LLM Model (from Flask)
            "llm_system_instruction": llm_system_instruction,  # Combined & dynamic LLM instruction
            "initial_greeting": initial_greeting,  # Initial greeting (dynamic with name)
            "campaign_design_prompt": agent_data['parameters']['campaignDesignPrompt'],  # 12. Campaign Design
        }
        logger.info(
            f"Successfully fetched and parsed agent config for {deployment_id}. Initial greeting: \"{initial_greeting}\"")
        return pipecat_config

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Flask backend or received error for {deployment_id}: {e}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from Flask backend: {e}")
        raise
    except KeyError as e:
        logger.error(
            f"Missing expected key in agent data from Flask backend: {e}. Check Flask API response structure and ensure all required fields are present.")
        raise
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise

async def main():
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily video transport with specific audio parameters
    - Gemini Live multimodal model integration
    - Voice activity detection
    - Animation processing
    - RTVI event handling
    """
    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        global _agent_deployment_id  # Declare intent to use the global variable

        # Retrieve the deployment ID from the global variable
        
        # deployment_id = "d18e401c-ca70-40da-b6a6-966d904a45f4"

        if deployment_id is None:
            logger.error("Agent Deployment ID was not set. Cannot start agent.")
            return  # Exit the coroutine if ID is missing

        deployment_id = _agent_deployment_id

        logger.info(f"Starting bot for agent deployment ID: {deployment_id}")

        agent_config = fetch_agent_config(deployment_id)
        keyjson_path = os.path.join(os.getcwd(), "key.json")

        stt = GoogleSTTService(
            credentials_path=keyjson_path,
            params=GoogleSTTService.InputParams(
                languages=[agent_config["language_enum"]],
                model="latest_long",
                enable_automatic_punctuation=True,
                enable_interim_results=True,
                enable_ssml=True,
                pitch="+2st",
                rate="1.2",
                volume="loud",
                emphasis="moderate"
            )
        )

        tts = GoogleTTSService(
            credentials_path=keyjson_path,
            voice_id=agent_config["voice_id"],
            audio_config_params=agent_config["speaking_rate"],
            params=GoogleTTSService.InputParams(
                language=agent_config["language_enum"],
                gender=agent_config["gender"],
            )
        )

        ssml_instruction_part = (
            "Your responses must also be formatted using SSML (Speech Synthesis Markup Language) "
            "within a <speak> tag. Use SSML tags like <break time='500ms'/> for pauses, "
            "<prosody rate='slow' pitch='high'> for voice adjustments, and "
            "<emphasis level='moderate'> for emphasis. "
            "For example: <speak>Hello! <break time='500ms'/> <emphasis level='strong'>How</emphasis> can I help you today?</speak>"
        )

        combined_system_instruction = f"{agent_config['llm_system_instruction']}\n\n{ssml_instruction_part}"

        # Set up Daily transport with specific audio/video parameters for Gemini
        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_out_enabled=True,
                video_out_width=1024,
                video_out_height=576,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            ),
        )

        # Initialize the Gemini Multimodal Live model
        llm = GoogleLLMService(
            api_key=os.getenv("GEMINI_API_KEY"),
            model=agent_config["llm_model"],
            system_instruction=agent_config["llm_system_instruction"],
        )

        messages = [
            {
                "role": "system",
                "content": agent_config["llm_system_instruction"],
            },
        ]

        # Set up conversation context and management
        # The context_aggregator will automatically collect conversation context
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        ta = TalkingAnimation()

        #
        # RTVI events for Pipecat client UI
        #
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                rtvi,
                context_aggregator.user(),
                llm,
                tts,
                ta,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )
        await task.queue_frame(quiet_frame)

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()
            # Kick off the conversation
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            print(f"Participant joined: {participant}")

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"Participant left: {participant}")
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start Gemini Bot Agent")
    parser.add_argument("--deployment_id", type=str, help="Deployment ID for agent config", required=False)
    parser.add_argument("-u", "--room_url", type=str, help="Daily room URL", required=False)
    parser.add_argument("-t", "--token", type=str, help="Daily room token", required=False)

    args = parser.parse_args()

    if args.deployment_id:
        set_deployment_id(args.deployment_id)
    else:
        print("No deployment_id provided — using default or none.")

        
    asyncio.run(main())
