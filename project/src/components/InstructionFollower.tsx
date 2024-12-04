import React, { useState } from "react";
import { Button } from "./ui/button";
const BASE_URL = "http://localhost:8000"; // Backend URL

const InstructionFollower: React.FC = () => {
  const [recordingStatus, setRecordingStatus] = useState<string>("Idle");
  const [transcription, setTranscription] = useState<string>("");
  const [generatedPrompt, setGeneratedPrompt] = useState<string>("");
  const [generatedImage, setGeneratedImage] = useState<string>("");
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [selectedOption, setSelectedOption] = useState<string>("1");
  let localAudioChunks: Blob[] = [];

  // Handlers
  const startRecording = async () => {
    try {
      console.log("Attempting to access the microphone...");
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  
      // Check if stream is valid
      if (!stream) {
        console.error("Failed to get user media stream.");
        setRecordingStatus("Failed to access microphone.");
        return;
      }
  
      console.log("Stream:", stream);
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });
  
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          localAudioChunks.push(event.data);
          console.log("Audio chunk added:", event.data.size);
        } else {
          console.warn("Empty audio chunk received.");
        }
      };
  
      recorder.onstart = () => {
        console.log("Recording started.");
        setRecordingStatus("Recording...");
      };  
  
      recorder.onstop = () => {
        console.log("Recording stopped. Audio chunks:", localAudioChunks);
        sendAudioToServer();
      };
  
      recorder.onerror = (err) => {
        console.error("Recorder error:", err);
        setRecordingStatus("Recording failed.");
      };
  
      recorder.start();
      setMediaRecorder(recorder);
      console.log("Recording started successfully.");
    } catch (error) {
      console.error("Error starting recording:", error);
      setRecordingStatus("Failed to start recording.");
    }
  };
  

  const stopRecording = () => {
    if (mediaRecorder) {
      mediaRecorder.requestData();
      mediaRecorder.stop();
      setRecordingStatus("Recording stopped.");
    }
  };

  const sendAudioToServer = async () => {
    const audioBlob = new Blob(localAudioChunks, { type: "audio/webm" });
    console.log("Blob size:", audioBlob.size);
  
    if (audioBlob.size === 0) {
      console.error("Recording is empty.");
      setRecordingStatus("Recording failed.");
      return;
    }
    const reader = new FileReader();
    reader.readAsDataURL(audioBlob);
    reader.onloadend = async () => {
      try {
        const base64Audio = reader.result;
        const response = await fetch(`${BASE_URL}/transcribe`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ audio_data: base64Audio }),
        });
  
        const data = await response.json();
        if (data.text) {
          setTranscription(data.text);
          setRecordingStatus("Transcription completed.");
        } else {
          console.error("Transcription failed:", data);
        }
      } catch (error) {
        console.error("Error sending audio to server:", error);
      }
    };
  
    reader.onerror = () => {
      console.error("Error reading audio blob.");
    };
  };
  

  const generatePrompt = async () => {
    try {
      const response = await fetch(`${BASE_URL}/generate_prompt2`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: transcription }),
      });

      const data = await response.json();
      setGeneratedPrompt(data.prompt);
    } catch (error) {
      console.error("Error generating prompt:", error);
    }
  };

  const generateImage = async () => {
    const endpoint = selectedOption === "1" ? "/generate_image1" : "/generate_image2";
    try {
      const response = await fetch(`${BASE_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: generatedPrompt }),
      });
      const data = await response.json();
      setGeneratedImage(`data:image/jpeg;base64,${data.image_base64}`);
      console.log("Generated Image URL:", generatedImage);
    } catch (error) {
      console.error("Error generating image:", error);
    }
  };

  return (
    <div className="border border-white rounded-md p-4">
      <div className="grid grid-cols-2 gap-5">
        {/* Recording Section */}
        <div className="grid col-span-1 gap-5 bg-custom_1 p-4 rounded-md">
          <h3 className="scroll-m-20 text-2xl font-semibold tracking-tight">
            Speak your prompt
          </h3>
          <Button onClick={startRecording}>
            <p className="text-center">START RECORDING</p>
          </Button>
          <Button onClick={stopRecording}>
            <p className="text-center">STOP RECORDING</p>
          </Button>
          <p className="text-sm mt-2">{recordingStatus}</p>
          <p className="text-sm mt-2">Transcription: {transcription}</p>
          <div className="flex flex-col gap-5">
            <Button onClick={generatePrompt}>
              <p className="text-center">GENERATE IMAGE PROMPT</p></Button>
              <p className="text-sm mt-2">Prompt Generated: {generatedPrompt}</p>
              <div className="flex flex-row items-center gap-10 mt-3">
              <label>
                <input
                  type="radio"
                  name="imageOption"
                  value="1"
                  checked={selectedOption === "1"}
                  onChange={() => setSelectedOption("1")}
                />
                 Pollination AI
              </label>
              <label>
                <input
                  type="radio"
                  name="imageOption"
                  value="2"
                  checked={selectedOption === "2"}
                  onChange={() => setSelectedOption("2")}
                />
                 Flux
              </label>
            </div>
            <Button onClick={generateImage}>
              <p className="text-center">GENERATE IMAGE</p></Button>
          </div>
        </div>

        {/* Generated Image Section */}
        <div className="grid col-span-1 gap-5 bg-custom_1 p-4 rounded-md">
          <h3 className="scroll-m-20 text-2xl font-semibold tracking-tight">
            Generated Image
          </h3>
          {generatedImage ? (
            <img
              src={generatedImage}
              alt="Generated"
              className="rounded-md"
              style={{ maxWidth: "100%" }}
            />
          ) : (
            <p>No image generated yet.</p>
          )}
        </div>
      </div>
    </div>
  );
};

export default InstructionFollower;
