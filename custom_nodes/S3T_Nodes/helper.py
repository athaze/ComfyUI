
# helper.py
import torch
from llama_cpp import Llama
import gc
from omnivoice import OmniVoice
import soundfile as sf
import logging
import time 
from .UserDatabase import Usagereporter
import os 
# from nemo.collections.asr.models import SortformerEncLabelModel
from .indic_transcribe import IndicTranscribe
from .long_form import transcribe_long

# [] change models to prov varients  

# []maybe use speaker dir then split audio into timestamps pass each 
#   batch and transcribe then stich each audio with its duration so the lip syncing can be slightly more better
       # [x] Speeker Dir Cuting Audio 
       # [ ] Passing Batch lip  

if not torch.cuda.is_available():
    raise RuntimeError("CUDA/GPU is required but not available.")

logging.basicConfig(level=logging.INFO,format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger=logging.getLogger("helper.Logging")

## TODO pass hf_token in each models loading for private access 
cuda = "cuda"

model=None

def transcribe_audio(audio_file): #audio filepath post
    global model
    if model is None:
        logger.info("Proccesing Stage One  ")
        try:
            model=IndicTranscribe.from_pretrained(model_dir)

            logger.info("Dictation Model Loaded")

        except Exception as e:
            logger.error("Check the Audio file is less than 30 minutes ")
            raise e
        
        logger.info(f"Processing NodeOne {model.device} ") 
        
    start_time=time.time()
    
    try:
        transcription=transcribe_long(asr,audio_file,lang="None")
        transcription=transcription[0].text
    except Exception as e:
        raise RuntimeError(e)

    finally:
            if os.path.exists(audio_file):
                os.remove(audio_file)
                            
    elapsed_time=time.time()-start_time # DEBUG

    logger.info(f"transcription completed Took {elapsed_time:.2f}")


    return transcription # Node passing 


llm=None
# [ ]TODO: Return the token usage

def translate_text(text:str,target_lang:str): # text and target lang POST
    
    global llm 

    if llm is None:
        
        llm = Llama(
            model_path="Gemma/gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf",# Abs path 
            n_ctx=4096, 
            n_gpu_layers=-1,           # use all layers on GPU (set to 0 for CPU-only)
            verbose=False
        )
    logger.info("Translation Ready to Process")
    messages = [
    {"role": "system",
            "content": """**Role:** You are a Dual-Expert AI Agent: a Master Translator and a Senior Linguistic Editor.

                    **Objective:** Your goal is to transform the provided [Source Text] into a [Polished Translation] in the [Target Language] through a strict two-stage internal process.
                    
                    **INTERNAL WORKFLOW (Do not output these steps):**
                    
                    **STAGE 1: High-Fidelity Translation**
                    - Analyze the [Source Text] for exact meaning, nuance, and intent.
                    - Generate a direct, high-accuracy translation into the [Target Language].
                    - Ensure no facts, names, or numbers are altered or omitted.
                    
                    **STAGE 2: Professional Post-Editing (Cleanup)**
                    - Take the Stage 1 draft and perform a deep linguistic cleanup:
                    1. **Fluency:** Eliminate "translationese" and word-for-word phrasing. Reconstruct sentences to follow the natural, idiomatic syntax of the [Target Language].
                    2. **Tone & Register:** Match the tone of the source (e.g., if the source is a formal speech, the output must be sophisticated and authoritative; if it is news, it must be journalistic).
                    3. **Lexical Precision:** Replace generic or literal words with contextually powerful and professional vocabulary.
                    4. **Mechanical Perfection:** Correct all spelling, grammar, punctuation, and capitalization errors.
                    5. **Rhetorical Flow:** Ensure the text has a natural rhythm, using appropriate pauses (punctuation) for the target reader.
    

                    **CRITICAL OUTPUT INSTRUCTION:**
                    - Do NOT show your internal reasoning.
                    - Do NOT show the "Stage 1" draft.
                    - Do NOT include any introductory remarks (e.g., "Here is the translation...").
                    - Provide ONLY the final, result from STAGE 2 ."""},

    {"role": "user",
    "content": f"{text} <translate> to <{target_lang}>"}]
    logger.info("Sentence translating ")

    start_time= time.time()


    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=None,
        temperature=0.1, # .2 for gemma 
        top_p=0.5, # 0.5 for gemma 
        top_k=15)

    elapsed_time=time.time()-start_time
    logger.info(f"Translation completed Took {elapsed_time:.2f}")
    translation = response['choices'][0]['message']['content'] 

    logger.info("Translator Reseting ")
    
    llm.reset()

    logger.info("Flushed")

    torch.cuda.empty_cache() ## clear the cache of the genration 

    return translation

    
Speech=None

def Speech_generate(ref_audio,translated_text,ref_text,duration,language):
    logger.info("Beginning Speech Genaration ")

    global Speech , llm , model 

    if model is not None:
        model = None
        
    if llm is not None:
        llm = None
    
    gc.collect()

    torch.cuda.empty_cache()

    logger.info("Flushed")

    logger.info("Beginning Node.Speech ")

    if Speech is None:
        Speech = OmniVoice.from_pretrained(
        "athaze/OmniVoice",
        device_map=cuda,
        dtype=torch.float16
        )
    else:
        logger.info("Speech Loaded already")

    # Generate audio
    start_time=time.time()
    logger.info("Starting Speech Genaration this might take a while")

    audio = Speech.generate(
        text=translated_text,
        ref_audio=ref_audio,
        ref_text=ref_text,
        duration=duration,
        postprocess_output=False,
        language = language)
    
    elapsed_time=time.time()-start_time

    actual_sec = audio[0].shape[-1] / 24000
    Reporter = Usagereporter()
    Reporter.CredentialLogin()
    Reporter.CurrentUsage()
    Reporter.UpdateUsage(actual_sec)


    logger.info(f"Speech Completed took {elapsed_time:.2f}")
    
    return audio[0]

