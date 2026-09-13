
#nodes.py

from .helper import transcribe_audio , translate_text , Speech_generate 
import os 
import torchaudio, tempfile
from comfy_api.latest import ComfyExtension, io, ui
import torch
import soundfile as sf

class Speech_To_Text(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Speech_to_text_V0.1",
            display_name="Transcribe_speech",
            category="INPROV/Transcribe",
            inputs=[
                    io.Audio.Input("audio_file"),],
            outputs=[
                io.String.Output("transcription"),
                io.Float.Output("duration")])
    
    # call the transsribe then call the translate 
    @classmethod
    def execute(cls, audio_file):
        waveform = audio_file["waveform"]
        sr = audio_file["sample_rate"]
    
        tmp_path = None
        
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                torchaudio.save(tmp_path, waveform.squeeze(0), sr)
                
                transcription = transcribe_audio(tmp_path) # Got it 
                
                duration = sf.info(tmp_path).duration
                
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
        return io.NodeOutput(transcription,duration)
            
class Text_to_Text(io.ComfyNode):
    
    @classmethod
    def define_schema(cls) -> io.Schema:
    
        return io.Schema(
            node_id="Translation",
            display_name="Translate_Speech",
            category="INPROV/Translate",
            inputs=[
                    io.String.Input("transcription"),
                    io.Combo.Input(
                    "language",
                    options=["Malayalam","English"],
                    default="Malayalam",)],
            outputs=[
                io.String.Output("translation")])
    @classmethod
    def execute(cls,transcription,language) -> io.NodeOutput:
        translation=translate_text(text=transcription,target_lang = language)
        
        return io.NodeOutput(translation,ui=ui.PreviewText(translation)) 


class Text_to_Speech(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TextToSpeech",
            display_name="Translated -> Speech",
            category="INPROV/Speech",
            inputs=[
                io.String.Input("Ttext"),
                io.Audio.Input("audio_file"),
                io.String.Input("Rtext"),
                io.Float.Input("duration"),
                io.Combo.Input("language",options=["Malayalam","English",],default="Malayalam")
            ],
            is_output_node=True,
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo], 
            outputs=[
                io.Audio.Output("speech")])

    @classmethod
    def execute(cls,Ttext,audio_file,Rtext,duration,language):
        ## Incoming Reference Audio 
        waveform = audio_file["waveform"]
        sr = audio_file["sample_rate"]

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                torchaudio.save(tmp.name, waveform.squeeze(0), sr)
                tmp_path = tmp.name
                # Genarated Speech Audio Output
                arr = Speech_generate(ref_audio=tmp_path,translated_text=Ttext,ref_text=Rtext,duration=duration,language=language)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path) 
            
        waveform = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [samples] -> [1, 1, samples]
        result = {"waveform": waveform, "sample_rate":24000}

        return io.NodeOutput(result,ui=ui.PreviewAudio(result,cls=cls))
        

class Dub(io.ComfyNode):
    @classmethod 
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SpeechToSpeech",
            display_name="Dubing ",
            category="INPROV/Dub",
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio_file"),
                io.Combo.Input("language",options=["Malayalam","English",],default="Malayalam"),])
    @classmethod
    def execute(cls,audio_file,language):
        waveform = audio_file["waveform"]
        sr = audio_file["sample_rate"]
    
        tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                torchaudio.save(tmp.name, waveform.squeeze(0), sr)
                tmp_path=tmp.name
                duration = sf.info(tmp_path).duration

                transcription = transcribe_audio(tmp_path) ## use split for Speech to text 
                translation=translate_text(text=transcription,target_lang = language) # Text to Text 
                arr = Speech_generate(ref_audio=tmp.name,translated_text=translation,ref_text=transcription,duration=duration,language=language) # Text to Speech
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        waveform = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        result = {"waveform": waveform, "sample_rate":24000}


        return io.NodeOutput(result,ui=ui.PreviewAudio(result,cls=cls))







        
        