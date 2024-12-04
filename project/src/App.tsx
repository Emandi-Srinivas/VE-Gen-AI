import './App.css';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./components/ui/tabs";
import ImageGenerator from './components/ImageGenerator';
import VideoGenerator from './components/VideoGenerator';
import InstructionFollower from './components/InstructionFollower';
import LanguageModal from './components/LanguageModel';
import { useState, useEffect } from 'react';

function App() {
  const [language, setLanguage] = useState<string | undefined>(undefined);
  const [isLoading, setIsLoading] = useState(false);

  const handleLanguageSelect = async (selectedLanguage: string) => {
    console.log('Language selected:', selectedLanguage); // Debug log
    try {
      setIsLoading(true);
      setLanguage(selectedLanguage);
    } catch (error) {
      console.error('Error setting language:', error);
      setLanguage(undefined);
    } finally {
      setIsLoading(false);
    }
  };

  // Debug log to track language state changes
  useEffect(() => {
    console.log('Current language state:', language);
  }, [language]);

  if (isLoading) {
    return (
      <div className="bg-black min-h-screen flex items-center justify-center">
        <div className="text-white text-xl">Loading...</div>
      </div>
    );
  }

  return (
    <div className='bg-black min-h-screen'>
      {!language && (
        <div className="fixed inset-0 bg-black flex flex-col justify-center items-center">
          <h1 className="scroll-m-20 text-4xl font-extrabold tracking-tight lg:text-5xl text-white mb-8 drop-shadow-lg">
            VE - Gen AI
          </h1>
          <LanguageModal onLanguageSelect={handleLanguageSelect} />
        </div>
      )}
      
      {language && (
        <>
          <Navbar />
          <div className='m-5'>
            <h1 className="scroll-m-20 text-center text-4xl text-white font-extrabold tracking-tight lg:text-5xl mb-4">
              VE - Gen AI
            </h1>
            <p className="text-center text-white mt-2 mb-6">Selected Language: {language}</p>
            <div className='flex justify-center items-center my-5'>
              <Tabs defaultValue="ImageGenerator" className="w-full mx-auto bg-black text-white p-5 flex flex-col border-gray-500 border-2 rounded-lg">
                <TabsList className='bg-custom_1 gap-5 mb-4'>
                  <TabsTrigger value="ImageGenerator">Image Generator</TabsTrigger>
                  <TabsTrigger value="VideoGenerator">Video Generator</TabsTrigger>
                  <TabsTrigger value="Instruction">Instruction Follower</TabsTrigger>
                </TabsList>
                <TabsContent value="ImageGenerator">
                  <ImageGenerator />
                </TabsContent>
                <TabsContent value="VideoGenerator">
                  <VideoGenerator />
                </TabsContent>
                <TabsContent value="Instruction">
                  <InstructionFollower />
                </TabsContent>
              </Tabs>
            </div>
          </div>
          <Footer />
        </>
      )}
    </div>
  );
}

export default App;