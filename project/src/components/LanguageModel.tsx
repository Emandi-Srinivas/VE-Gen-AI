import React, { useState } from 'react';

interface LanguageModalProps {
  onLanguageSelect: (language: string) => void;
}

const LanguageModal: React.FC<LanguageModalProps> = ({ onLanguageSelect }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedButton, setSelectedButton] = useState<string | null>(null);

  const handleLanguageSelect = async (language: string) => {
    setIsLoading(true);
    setError(null);
    setSelectedButton(language);
    
    try {
      const response = await fetch('http://localhost:8000/set_language/', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ language }),
      });

      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to set language');
      }

      console.log('Language set successfully:', data);
      onLanguageSelect(language);
    } catch (err) {
      console.error('Error in language selection:', err);
      setError(err instanceof Error ? err.message : 'An error occurred while setting the language');
    } finally {
      setIsLoading(false);
      setSelectedButton(null);
    }
  };

  const ButtonContent = ({ isCurrentButton, language }: { isCurrentButton: boolean, language: string }) => (
    isLoading && isCurrentButton ? (
      <span className="flex items-center justify-center">
        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
        Loading Model...
      </span>
    ) : language
  );

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex justify-center items-center z-50">
      <div className="bg-white rounded-lg shadow-lg p-6 max-w-sm w-full mx-4">
        <h2 className="text-xl font-bold mb-4">Select Your Language</h2>
        <p className="text-gray-700 mb-6">Choose a language to load the appropriate model.</p>
        
        {error && (
          <div className="mb-4 p-3 bg-red-100 text-red-700 rounded-md text-sm">
            {error}
          </div>
        )}

        <div className="flex flex-wrap gap-4 justify-center">
          <button
            className={`bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-6 rounded transition-colors duration-200 min-w-[120px] ${
              isLoading && selectedButton !== 'Telugu' ? 'opacity-50 cursor-not-allowed' : ''
            }`}
            onClick={() => handleLanguageSelect('Telugu')}
            disabled={isLoading}
          >
            <ButtonContent isCurrentButton={selectedButton === 'Telugu'} language="Telugu" />
          </button>
          <button
            className={`bg-green-500 hover:bg-green-600 text-white font-semibold py-2 px-6 rounded transition-colors duration-200 min-w-[120px] ${
              isLoading && selectedButton !== 'Hindi' ? 'opacity-50 cursor-not-allowed' : ''
            }`}
            onClick={() => handleLanguageSelect('Hindi')}
            disabled={isLoading}
          >
            <ButtonContent isCurrentButton={selectedButton === 'Hindi'} language="Hindi" />
          </button>
        </div>
      </div>
    </div>
  );
};

export default LanguageModal;