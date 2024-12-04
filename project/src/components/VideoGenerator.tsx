import { Button } from "./ui/button";
import { useState } from "react";
const BASE_URL = "http://localhost:8000"; // Backend URL

const VideoGenerator = () => {
  const [generatedVideo, setGeneratedVideo] = useState<string>("");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [selectedOption, setSelectedOption] = useState<string>("1");

  const handleAudioUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]; // Optional chaining to safely access the file
    if (file && file.type.startsWith("audio/")) {
      setAudioFile(file);
      console.log("Audio file uploaded:", file.name);
    } else {
      alert("Please upload a valid audio file.");
    }
  };


  const generateVideo = async () => {
    if (!audioFile) {
      alert("Please upload an audio file first!");
      return;
    }
    const formData = new FormData();
    formData.append("audio", audioFile);  // Make sure the key matches the backend's expected key
    const endpoint = selectedOption === "1" ? "/generate_video1" : "/generate_video2";
    try {
      const response = await fetch(`${BASE_URL}${endpoint}`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error("Failed to generate video.");
      }
      const videoBlob = await response.blob();

      // Create a URL for the video blob
      const videoUrl = URL.createObjectURL(videoBlob);

      // Set the generated video URL to display
      setGeneratedVideo(videoUrl);
    } catch (error) {
      console.error("Error generating video:", error);
    }
  };


  return (
    <div className="border border-white rounded-md p-4">
      <div className="grid grid-cols-2 gap-5">
        {/* Audio Upload Section */}
        <div id="Upload Audio" className="grid col-span-1 gap-5 bg-custom_1 p-4 rounded-md">
          <div className="flex flex-col gap-5">
            <h3 className="scroll-m-20 text-2xl font-semibold tracking-tight">
              Upload Your Audio (.mp3)
            </h3>
            <input
              type="file"
              accept=".mp3"
              onChange={handleAudioUpload}
              className="file-input"
            />
            <div className="flex flex-row items-center gap-10 mt-3">
              <label>
                <input
                  type="radio"
                  name="imageOption"
                  value="1"
                  checked={selectedOption === "1"}
                  onChange={() => setSelectedOption("1")}
                />
                PollinationAI/Flux
              </label>
              <label>
                <input
                  type="radio"
                  name="imageOption"
                  value="2"
                  checked={selectedOption === "2"}
                  onChange={() => setSelectedOption("2")}
                />
                CharacterFactory
              </label>
            </div>
            <Button onClick={generateVideo}>
              <p className="text-center">GENERATE VIDEO</p>
            </Button>
          </div>
        </div>

        {/* Generated Video Section */}
        <div id="Generated Video" className="grid col-span-1 gap-5 bg-custom_1 p-4 rounded-md">
          <h3 className="scroll-m-20 text-2xl font-semibold tracking-tight">Generated Video</h3>
          {generatedVideo ? (
            <div>
              <h3>Generated Video</h3>
              <video controls className="rounded-md" style={{ maxWidth: "100%" }}>
                <source src={generatedVideo} type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
          ) : (
            <p>No video generated yet.</p>
          )}
        </div>

      </div>
    </div>
  );
};

export default VideoGenerator;