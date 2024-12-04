import base64
import os
import subprocess

def save_audio(audio_data):
    try:
        # Handle base64 decoding
        if "," in audio_data:
            audio_bytes = base64.b64decode(audio_data.split(",")[1])
        else:
            audio_bytes = base64.b64decode(audio_data)
        
        # Save the audio to a temporary file
        temp_file = "temp_audio.webm"
        with open(temp_file, "wb") as f:
            f.write(audio_bytes)
        
        print(f"Saved {temp_file}, size: {os.path.getsize(temp_file)} bytes")

        # Convert WebM to WAV using ffmpeg
        wav_file = "temp_audio.wav"
        if os.path.exists(wav_file):  # Remove existing file
            os.remove(wav_file)

        subprocess.run(
            ["ffmpeg", "-i", temp_file, wav_file],
            check=True,
            capture_output=True
        )
        print(f"Conversion successful: {wav_file}")
        os.remove(temp_file)  # Clean up temporary WebM file
        return wav_file

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode()}")
        raise ValueError("Audio conversion failed. Ensure ffmpeg is installed and the input is valid.")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise ValueError("Failed to process audio data.")
