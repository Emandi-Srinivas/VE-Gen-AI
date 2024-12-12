from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
from typing import List, Optional
from pydub import AudioSegment
import json
from moviepy.editor import *
from moviepy.video.tools.subtitles import SubtitlesClip
import base64
from io import BytesIO
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from huggingface_hub import InferenceClient
import speech_recognition as sr
import google.generativeai as genai
import random
import os
import urllib.parse
import time
import requests
import torch
from .utils import save_audio
from fastapi.middleware.cors import CORSMiddleware
from gradio_client import Client, handle_file
from transformers import pipeline, WhisperForConditionalGeneration, GenerationConfig, WhisperTokenizer, WhisperFeatureExtractor
from dotenv import load_dotenv
from PIL import Image
load_dotenv()

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

app = FastAPI()
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

genai.configure(api_key=os.environ["API_KEY"])
gemini_model = genai.GenerativeModel('gemini-1.5-pro')

device = "mps" if torch.backends.mps.is_available() else "cpu"

class LanguageRequest(BaseModel):
    language: str

class AudioData(BaseModel):
    audio_data: str

class ImagePrompt(BaseModel):
    prompt: str

class PromptRequest(BaseModel):
    text: str

model = None
tokenizer = None
feature_extractor = None
transcribe = None
font_path = None


def load_model(language: str):
    """
    Load model, tokenizer, and feature extractor based on the selected language.
    """
    global model, tokenizer, feature_extractor, transcribe, font_path

    if language.lower() == "telugu":
        model_name = "vasista22/whisper-telugu-medium"
        font_path = "/System/Library/Fonts/KohinoorTelugu.ttc"
        
    elif language.lower() == "hindi":
        # model_name = "vasista22/whisper-hindi-large-v2"
        model_name = "vasista22/whisper-hindi-medium"
        font_path = "/System/Library/Fonts/Kohinoor.ttc"
    else:
        raise ValueError("Unsupported language. Please choose 'Hindi' or 'Telugu'.")

    # Load the model, tokenizer, and feature extractor
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

    # Configure generation settings
    generation_config = GenerationConfig.from_pretrained("openai/whisper-base")
    model.generation_config = generation_config

    # Initialize the pipeline
    transcribe = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        device=device
    )

# API route to update the model based on user input
@app.post("/set_language/")
async def set_language(request: LanguageRequest):
    try:
        load_model(request.language)
        return {"message": f"Model switched to {request.language} successfully!"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# @app.get("/")
# async def read_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe")
async def transcribe_audio(audio_data: AudioData):
    try:
        # Save the audio data to a file
        audio_file = save_audio(audio_data.audio_data)

        # Transcribe the audio
        result = transcribe(audio_file)["text"]

        return JSONResponse(content={"text": result})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/generate_prompt1")
async def generate_prompt(prompt_request: PromptRequest):
    try:
        text = prompt_request.text
        history = []
        chat = gemini_model.start_chat(history=history)
        prompttext = f'''
        This is a programmatically generated input. You are provided with a text in regional indic language. Your tasks are :
        1) Translate the regional indic language text into english text.
        2) Do not add any extra information like introduction, notes, or explanations—just the translated text.
        Here is the input:
        {text}'''
        response1 = chat.send_message(prompttext)
        for candidate in response1.candidates:
            for part in candidate.content.parts:
                text = part.text
                print(part.text)
        chat = gemini_model.start_chat(history=history)
        prompttext = f'''
        This is a programmatically generated input. You are provided with a text. Your tasks are :
        1) Generate an Image Prompt which clearly describes the text visually in an image.
        2) Do not add any extra information like introduction, notes, or explanations—just the image prompt text.
        Here is the input:
        {text}'''
        response1 = chat.send_message(prompttext)
        for candidate in response1.candidates:
            for part in candidate.content.parts:
                text = part.text
                print(part.text)
        return JSONResponse(content={"prompt": text})
    except Exception as e:
        print(f"Error generating prompt: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    

@app.post("/generate_prompt2")
async def generate_prompt(prompt_request: PromptRequest):
    try:
        text = prompt_request.text
        history = []
        chat = gemini_model.start_chat(history=history)
        prompttext = f'''
        This is a programmatically generated input. You are provided with a text in regional indic language. Your tasks are :
        1) Translate the regional indic language text into english text.
        2) Do not add any extra information like introduction, notes, or explanations—just the translated text.
        Here is the input:
        {text}'''
        response1 = chat.send_message(prompttext)
        for candidate in response1.candidates:
            for part in candidate.content.parts:
                text = part.text
                print(part.text)
        chat = gemini_model.start_chat(history=history)
        prompttext = f'''
        This is a programmatically generated input. You are provided with an instruction. Your tasks are :
        1) Generate an Image Prompt which clearly describes the instruction visually in an image.
        2) Do not add any extra information like introduction, notes, or explanations—just the image prompt text.
        Here is the input:
        {text}'''
        response1 = chat.send_message(prompttext)
        for candidate in response1.candidates:
            for part in candidate.content.parts:
                text = part.text
                print(part.text)
        return JSONResponse(content={"prompt": text})
    except Exception as e:
        print(f"Error generating prompt: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
def download_image(image_url, file_name, retries=6):
    for attempt in range(retries):
        try:
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            with open(file_name, 'wb') as f:
                f.write(response.content)
            print(f"Image saved as: {file_name}")
            return response.content  # Return the image content directly
        except requests.exceptions.RequestException as e:
            print(f'Error downloading image: {e}. Attempt {attempt + 1} of {retries}.')
            time.sleep(5)  # Wait before retrying
    raise Exception("Failed to download image after several attempts.")
    

@app.post("/generate_image1")
async def generate_image(image_prompt: dict):
    try:
        prompt = image_prompt.get("prompt")
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")

        print(f"Received prompt: {prompt}")

        # Encode the prompt for URL safety
        encoded_prompt = urllib.parse.quote(prompt)
        # Generate the image URL for Pollinations AI
        image_url = f"https://pollinations.ai/p/{encoded_prompt}?width=1280&height=1280&seed=-1&model=flux&nologo='true'"
        
        # Define a filename for saving the image
        file_name = f"generated_image_{int(time.time())}.jpg"

        # Download and save the image
        image_content = download_image(image_url, file_name)

        # Convert the image content to a base64 string
        buffered = BytesIO()
        buffered.write(image_content)
        buffered.seek(0)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Return the base64 image and filename
        return JSONResponse(content={"image_base64": img_str, "filename": file_name})
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/generate_image2")
async def generate_image(image_prompt: ImagePrompt):
    try:
        prompt = image_prompt.prompt
        print(f"Received prompt: {prompt}")

        # Encode the prompt for URL safety
        encoded_prompt = urllib.parse.quote(prompt)
        random_seed = random.randint(0, 2**16 - 1)
        print(f"Using seed: {random_seed}")
        
        # Initialize the client with the specified model and token
        client = InferenceClient("black-forest-labs/FLUX.1-schnell", token=)
        # Generate the image without specifying optional parameters
        image = client.text_to_image(
            prompt=encoded_prompt,
            width=1024, 
            height=1024,
            seed=random_seed
        )
        
        # Save the image to a file in the backend
        image_name = f"generated_image_{int(time.time())}.jpg"
        image.save(image_name, format="JPEG")
        print(f"Image saved as: {image_name}")
        
        # Convert the image to a base64 string
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        return JSONResponse(content={"image_base64": img_str, "filename": image_name})
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))



def generate_images_CC(prompts_data):
    output = "\n".join([prompt["prompt_text"] for prompt in prompts_data["prompts"]])
    client = Client("http://127.0.0.1:7860/")

    # Make a prediction
    result = client.predict(
        chose_emb=handle_file('https://github.com/gradio-app/gradio/raw/main/test/test_files/sample_file.pdf'),
        choice="Create a new character",
        gender_GAN="Man",
        prompts_array=output,
        api_name="/generate_image"
    )
     # Extract image paths from the result
    images = result[0]  # This should contain the image details
    
    for i, img_info in enumerate(images):
        image_path = img_info['image']
        with Image.open(image_path) as img:
            output_path = f"image_{i+1}.jpg"
            img.convert("RGB").save(output_path, "JPEG")
            print(f"Image saved: {output_path}")

def ms_to_timecode(ms):
    hours = ms // 3600000
    ms = ms % 3600000
    minutes = ms // 60000
    ms = ms % 60000
    seconds = ms // 1000
    milliseconds = ms % 1000
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def generate_images_for_prompt1(prompts_data):
    array = []

    for idx, prompt in enumerate(prompts_data["prompts"], start=1):
        sentence = prompt["prompt_text"]
        encoded_sentence = urllib.parse.quote(sentence)
        random_seed = random.randint(0, 2**16 - 1)
        print(random_seed)
        client = InferenceClient("black-forest-labs/FLUX.1-schnell", token=array[idx%3])
        image = client.text_to_image(
            prompt=encoded_sentence,
            width=1024,
            height=1024,
            seed=random_seed,
        )
        image_name = f"image_{idx}.jpg"
        print(f"image saved: {image_name}")
        image.save(image_name)
        time.sleep(3)


def generate_images_for_prompt2(prompts_data, width=1280, height=1280, seed=-1, model='flux-realism'):
    # Loop through each prompt in the parsed JSON structure
    for idx, prompt in enumerate(prompts_data['prompts'], start=1):
        sentence = prompt['prompt_text']  # Extract the prompt_text
        encoded_sentence = urllib.parse.quote(sentence)
        image_url = f"https://pollinations.ai/p/{encoded_sentence}?width={width}&height={height}&seed={seed}&model={model}&nologo=true"
        image_name = f"image_{idx}.jpg"
        download_image(image_url, image_name)
        time.sleep(5)

@app.post("/generate_video1")
async def generate_video(audio: UploadFile = File(...)):
    try:
        audio_path = f"uploaded_audio.mp3"
        with open(audio_path, "wb") as audio_file:
            audio_file.write(audio.file.read())
        print("audio uploaded")
        # Step 1: Process audio into chunks
        audio = AudioSegment.from_mp3(audio_path)
        chunk_length_ms = 5000
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        audio_len = len(audio) // 1000
        chunk_len = chunk_length_ms // 1000

        srt_subtitles = ""
        current_time_ms = 0

        for i, chunk in enumerate(chunks):
            chunk_name = f"chunk_{i}.wav"
            chunk.export(chunk_name, format="wav")
            result = transcribe(chunk_name)["text"]
            start_time_ms = current_time_ms
            end_time_ms = current_time_ms + len(chunk)
            start_time = ms_to_timecode(start_time_ms)
            end_time = ms_to_timecode(end_time_ms)
            srt_subtitles += f"{i+1}\n{start_time} --> {end_time}\n{result.strip()}\n\n"
            current_time_ms += len(chunk)
            os.remove(chunk_name)

        srt_file = "subtitles.srt"
        final_srt_file = "fsubtitles.srt"
        with open(srt_file, "w", encoding="utf-8") as f:
            f.write(srt_subtitles)
        history = []
        chat = gemini_model.start_chat(history=history)
        prompttext = f'''
        This is a programmatically generated input. You are provided with subtitles and their respective timestamps in a structured format. Your tasks are :
        1) For each subtitle segment, translate the given Telugu subtitle into English under the regional language subtitle.
        2) The output format should strictly return the exact pattern but with respective English Text translation added.
        
           subtitle number
           timestamp
           regional lanuage subtitle
           English subtitle
        
        For example:=
            1
            00:00:00,000 --> 00:00:05,000
            గ్రామం ఉదయాన్ని ఆహ్వానిస్తోంది
            The village welcomes the sunrise
            
            2
            00:00:05,000 --> 00:00:10,000
            మార్కెట్‌లో సందడితో భరితమైంది
            The market is bustling with activity

        3) Generate the sequence completely, make sure no Subtitle is missed in the Final generation.
        4) Do not add any extra information like introduction, notes, or explanations—just the formatted subtitles.
        Here is the input:
        {srt_subtitles}'''
        print("translation started")
        response1 = chat.send_message(prompttext)
        
        
        final_subtitles = ""
        for candidate in response1.candidates:
            for part in candidate.content.parts:
                text = part.text
                final_subtitles += text.strip()
                print(part.text)
        
        print("Saving subtitles to subtitles.srt...\n")
        with open(final_srt_file, "w", encoding="utf-8") as f:
            f.write(final_subtitles)
        
        history.append({"role": "user", "content": prompttext})
        history.append({"role": "assistant", "content": response1})
        print("Creating Prompts for Ai image generator...\n")
        
        time.sleep(2)
        
        prompttext = f'''This is a programmatically generated input. Perform the following tasks for the given input:
        
                    You are provided with subtitles and their respective timestamps in a structured format. Your tasks are:
                    1. Generate a detailed image prompt for each scene described by the subtitles. Focus on the characters, their appearance, key visual elements, and the environment.
                    Ensure the prompt covers all necessary details to create an accurate image for the scene.
                    2. Calculate the duration for each prompt based on the timestamps provided. Adjust durations proportionally if the total duration does not match the audio length ({audio_len} seconds).
                    3. Return the output in the following JSON format:
                    
                    {{
                      "prompts": [
                        {{
                          "subtitle_indices": [1, 2],
                          "prompt_text": "A serene sunrise over a peaceful village, with farmers in traditional clothing, some holding tools and preparing for the day. The sky is painted in warm hues, and fields stretch out in the background.",
                          "duration": 10
                        }},
                        {{
                          "subtitle_indices": [3],
                          "prompt_text": "A bustling market scene filled with vibrant, colorful stalls. People of various ages are negotiating and shopping, with bright textiles and fresh produce filling the stands. There’s a sense of activity and excitement.",
                          "duration": 20
                        }}
                      ]
                    }}
                    
                    Important Rules:
                    - The total duration of all prompts must equal {audio_len} seconds.
                    - Use the timestamps provided in the subtitles to calculate durations proportionally.
                    - Do not allow overlapping subtitles for prompts.
                    - Do not add any extra information like introduction, notes, or explanations — just the formatted prompts and durations.
        
                    Rules for Image Prompt Characteristics:
                    - Focus on specific visual details for each scene. Describe the setting, environment, and characters' appearances, actions, and emotions clearly.
                    - Replace character names with their respective name, age, and key traits.
                    - Specify relationships (father, mother, son, etc.) for clarity in every prompt.
                    
                    Use the following subtitles as the basis for the prompts:
                    {final_subtitles}'''
        
        
        prompts2 = ""
        response2 = chat.send_message(prompttext)
        for candidate in response2.candidates:
            for part in candidate.content.parts:
                text = part.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                prompts2 += text.strip()

        prompts_data = json.loads(prompts2)
        print(prompts_data)

        # Step 3: Generate images for the video
        print("generating images")
        generate_images_for_prompt2(prompts_data)

        # Step 4: Create video from images and audio
        print("video creation started")
        image_paths = [f"image_{i+1}.jpg" for i in range(len(prompts_data["prompts"]))]
        durations = [prompt["duration"] for prompt in prompts_data["prompts"]]
        clips = [ImageClip(img).set_duration(d) for img, d in zip(image_paths, durations)]

        concat_clip = concatenate_videoclips(clips, method="compose")
        audio = AudioFileClip(audio_path).subclip(0, concat_clip.duration)
        final_clip = concat_clip.set_audio(audio)
        subtitles = SubtitlesClip("fsubtitles.srt", lambda txt: TextClip(txt, font=font_path, fontsize=24, color='white', bg_color='black'))
        subtitles = subtitles.set_duration(final_clip.duration).set_position(('center', 'bottom'))
        final_video = CompositeVideoClip([final_clip, subtitles])

        video_name = f"generated_video_{int(time.time())}.mp4"
        final_video.write_videofile(video_name, fps=24, audio_codec="aac")

        # Cleanup
        os.remove(audio_path)
        os.remove(srt_file)
        # for img in image_paths:
        #     os.remove(img)
        video_file = open(video_name, "rb")  # Open the video file in binary mode
        return StreamingResponse(video_file, media_type="video/mp4")  

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    



@app.post("/generate_video2")
async def generate_video(audio: UploadFile = File(...)):
    try:
        audio_path = f"uploaded_audio.mp3"
        with open(audio_path, "wb") as audio_file:
            audio_file.write(audio.file.read())

        # Step 1: Process audio into chunks
        audio = AudioSegment.from_mp3(audio_path)
        chunk_length_ms = 5000
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
        audio_len = len(audio) // 1000
        chunk_len = chunk_length_ms // 1000

        srt_subtitles = ""
        current_time_ms = 0

        for i, chunk in enumerate(chunks):
            chunk_name = f"chunk_{i}.wav"
            chunk.export(chunk_name, format="wav")
            result = transcribe(chunk_name)["text"]
            start_time_ms = current_time_ms
            end_time_ms = current_time_ms + len(chunk)
            start_time = ms_to_timecode(start_time_ms)
            end_time = ms_to_timecode(end_time_ms)
            srt_subtitles += f"{i+1}\n{start_time} --> {end_time}\n{result.strip()}\n\n"
            current_time_ms += len(chunk)
            os.remove(chunk_name)

        srt_file = "subtitles.srt"
        final_srt_file = "fsubtitles.srt"
        with open(srt_file, "w", encoding="utf-8") as f:
            f.write(srt_subtitles)

        history = []
        chat = gemini_model.start_chat(history=history)
        prompttext = f'''
        This is a programmatically generated input. You are provided with subtitles and their respective timestamps in a structured format. Your tasks are :
        1) For each subtitle segment, translate the given Telugu subtitle into English under the regional language subtitle.
        2) The output format should strictly return the exact pattern but with respective English Text translation added.
        
           subtitle number
           timestamp
           regional lanuage subtitle
           English subtitle
        
        For example:=
            1
            00:00:00,000 --> 00:00:05,000
            గ్రామం ఉదయాన్ని ఆహ్వానిస్తోంది
            The village welcomes the sunrise
            
            2
            00:00:05,000 --> 00:00:10,000
            మార్కెట్‌లో సందడితో భరితమైంది
            The market is bustling with activity

        3) Generate the sequence completely, make sure no Subtitle is missed in the Final generation.
        4) Do not add any extra information like introduction, notes, or explanations—just the formatted subtitles.
        Here is the input:
        {srt_subtitles}'''
        
        response1 = chat.send_message(prompttext)
        
        
        final_subtitles = ""
        for candidate in response1.candidates:
            for part in candidate.content.parts:
                text = part.text
                final_subtitles += text.strip()
                print(part.text)
        
        print("Saving subtitles to fsubtitles.srt...\n")
        with open(final_srt_file, "w", encoding="utf-8") as f:
            f.write(final_subtitles)
        
        history.append({"role": "user", "content": prompttext})
        history.append({"role": "assistant", "content": response1})
        print("Creating Prompts for Ai image generator...\n")
        
        time.sleep(2)
        prompttext = f'''This is a programmatically generated input. Perform the following tasks for the given input:
        
                You are provided with subtitles and their respective timestamps in a structured format. Your tasks are:
                1. Generate a really short and precise image prompt for each scene described by the subtitles. Focus on the characters name, age and appearance.
                2. Calculate the duration for each prompt based on the timestamps provided. Adjust durations proportionally if the total duration does not match the audio length ({audio_len} seconds).
                3. Return the output in the following JSON format:
                
                {{
                  "prompts": [
                    {{
                      "subtitle_indices": [1, 2],
                      "prompt_text": "Jenny as a child in a modest outfit, sitting at a small desk with books, facing the camera, best quality, ultra high res",
                      "duration": 10
                    }},
                    {{
                      "subtitle_indices": [3],
                      "prompt_text": "Jenny as a young woman, wearing a simple outfit, studying hard at a library, facing the camera, best quality, ultra high res",
                      "duration": 20
                    }}
                }}
                
                Important Rules:
                - The total duration of all prompts must equal {audio_len} seconds.
                - Use the timestamps provided in the subtitles to calculate durations proportionally.
                - Do not allow overlapping subtitles for prompts.
                - Do not add any extra information like introduction, notes, or explanations — just the formatted prompts and durations.
                
                Rules for Image Prompt Characteristics:
                - Describe the environment, actions, and emotions clearly in very short and natural form.
                - Replace character names with their respective name, age, and key traits.
                - The main_character word has to be repeating in every prompt.
                
                Use the following subtitles as the basis for the prompts:
                {final_subtitles}'''
        
        
        prompts2 = ""
        response2 = chat.send_message(prompttext)
        for candidate in response2.candidates:
            for part in candidate.content.parts:
                text = part.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                prompts2 += text.strip()

        prompts_data = json.loads(prompts2)
        print(prompts_data)
        print("Generating images...")
        generate_images_CC(prompts_data)
        prompts_data = json.loads(prompts2)["prompts"]
        print("video creation started")
        image_paths = [f"image_{i+1}.jpg" for i in reversed(range(len(prompts_data)))]
        durations = [prompt["duration"] for prompt in prompts_data]
        clips = [ImageClip(img).set_duration(d) for img, d in zip(image_paths, durations)]

        concat_clip = concatenate_videoclips(clips, method="compose")
        audio = AudioFileClip(audio_path).subclip(0, concat_clip.duration)
        final_clip = concat_clip.set_audio(audio)

        subtitles = SubtitlesClip("fsubtitles.srt", lambda txt: TextClip(txt, font=font_path, fontsize=24, color='white', bg_color='black'))
        subtitles = subtitles.set_duration(final_clip.duration).set_position(('center', 'bottom'))
        final_video = CompositeVideoClip([final_clip, subtitles])

        video_name = f"generated_video_{int(time.time())}.mp4"
        final_video.write_videofile(video_name, fps=24, audio_codec="aac")

        # Cleanup
        os.remove(audio_path)
        os.remove(srt_file)
        # for img in image_paths:
        #     os.remove(img)
        video_file = open(video_name, "rb")  # Open the video file in binary mode
        return StreamingResponse(video_file, media_type="video/mp4")  

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
