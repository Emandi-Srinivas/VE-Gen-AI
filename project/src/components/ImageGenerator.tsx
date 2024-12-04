import { Button } from "./ui/button";
import React, { useState } from "react";

const BASE_URL = "http://localhost:8000"; // Backend URL

const ImageGenerator: React.FC = () => {
  const [recordingStatus, setRecordingStatus] = useState<string>("Idle");
  const [transcription, setTranscription] = useState<string>("");
  const [generatedPrompt, setGeneratedPrompt] = useState<string>("");
  const [generatedImage, setGeneratedImage] = useState<string>("");
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [optionalDescription, setOptionalDescription] = useState<string>("");
  const [selectedOption, setSelectedOption] = useState<string>("1");

  let localAudioChunks: Blob[] = [];

  // Handlers
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!stream) {
        setRecordingStatus("Failed to access microphone.");
        return;
      }

      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          localAudioChunks.push(event.data);
        }
      };

      recorder.onstart = () => setRecordingStatus("Recording...");
      recorder.onstop = () => sendAudioToServer();

      recorder.onerror = () => setRecordingStatus("Recording failed.");

      recorder.start();
      setMediaRecorder(recorder);
    } catch (error) {
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

    if (audioBlob.size === 0) {
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
        }
      } catch (error) {
        console.error("Error sending audio to server:", error);
      }
    };
  };

  const generatePrompt = async () => {
    try {
      const response = await fetch(`${BASE_URL}/generate_prompt1`, {
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
    const finalPrompt = optionalDescription
      ? `${generatedPrompt} ${optionalDescription}`
      : generatedPrompt;
    try {
      const response = await fetch(`${BASE_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: finalPrompt }),
      });

      const data = await response.json();
      setGeneratedImage(`data:image/jpeg;base64,${data.image_base64}`);
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
              <p className="text-center">GENERATE IMAGE PROMPT</p>
            </Button>
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
            <textarea
              className="mt-3 p-2 border rounded-md text-black"
              placeholder="Optional: Add extra details for the image prompt"
              value={optionalDescription}
              onChange={(e) => setOptionalDescription(e.target.value)}
            />
            <Button onClick={generateImage}>
              <p className="text-center">GENERATE IMAGE</p>
            </Button>
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

export default ImageGenerator;
