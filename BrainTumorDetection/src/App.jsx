import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import axios from 'axios';

const App = () => {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  // Handle file drop/upload
  const handleDrop = (e) => {
    e.preventDefault();
    const uploadedFile = e.dataTransfer.files[0];
    processFile(uploadedFile);
  };

  const handleFileChange = (e) => {
    const uploadedFile = e.target.files[0];
    processFile(uploadedFile);
  };

  const processFile = (file) => {
    if (!file) return;

    setError(null);
    setResult(null);
    setFile(file);

    // Generate preview
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result);
    reader.readAsDataURL(file);
  };

  // Submit to backend
  const handleSubmit = async () => {
    if (!file) {
      setError("Please upload an image first!");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('image', file);

      const response = await axios.post('http://localhost:3001/api/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        timeout: 30000 // 30 seconds timeout
      });

      if (response.data.error) {
        throw new Error(response.data.error);
      }

      setResult({
        classification: response.data.result,
        confidence: response.data.confidence,
        details: response.data.details
      });
    } catch (err) {
      console.error('Frontend error:', err);
      setError(err.response?.data?.error ||
        err.response?.data?.message ||
        err.message ||
        "Failed to analyze image. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Reset
  const handleReset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-blue-900 text-white p-6">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          initial={{ y: -50, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h1 className="text-5xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-600">
            Brain Tumor Detection
          </h1>
          <p className="text-xl text-gray-300">
            Upload a brain scan MRI to detect tumors using AI.
          </p>
        </motion.div>

        {/* Upload Card */}
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="bg-gray-800/50 backdrop-blur-lg rounded-xl p-8 shadow-2xl border border-gray-700 mb-10"
        >
          <div
            className={`border-2 border-dashed rounded-lg p-12 text-center transition-all duration-300 ${!file ? 'border-cyan-400 hover:border-cyan-300 bg-gray-900/30' : 'border-transparent'}`}
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileInputRef.current.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              className="hidden"
              accept="image/*"
            />

            {!preview ? (
              <motion.div
                initial={{ opacity: 0.6 }}
                animate={{ opacity: 1 }}
                transition={{ repeat: Infinity, repeatType: "reverse", duration: 1.5 }}
              >
                <div className="flex flex-col items-center justify-center space-y-4">
                  <svg className="w-16 h-16 text-cyan-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                  </svg>
                  <p className="text-xl">Drag & drop an MRI scan or click to browse</p>
                  <p className="text-gray-400">Supports: JPG, PNG, DICOM</p>
                </div>
              </motion.div>
            ) : (
              <motion.div
                initial={{ scale: 0.95 }}
                animate={{ scale: 1 }}
                className="relative"
              >
                <img
                  src={preview}
                  alt="Preview"
                  className="max-h-64 mx-auto rounded-lg shadow-lg border border-gray-600"
                />
                <button
                  onClick={(e) => { e.stopPropagation(); handleReset(); }}
                  className="absolute -top-3 -right-3 bg-red-500 hover:bg-red-600 text-white p-2 rounded-full shadow-lg transition-all"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </motion.div>
            )}
          </div>

          <div className="flex justify-center mt-6">
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={handleSubmit}
              disabled={isLoading || !file}
              className={`px-8 py-3 rounded-full font-bold text-lg flex items-center space-x-2 ${isLoading || !file ? 'bg-gray-600 cursor-not-allowed' : 'bg-cyan-600 hover:bg-cyan-700 shadow-lg shadow-cyan-500/20'}`}
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing...
                </>
              ) : (
                <>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
                  </svg>
                  <span>Detect Tumor</span>
                </>
              )}
            </motion.button>
          </div>
        </motion.div>

        {/* Results */}
        <AnimatePresence>
          {(result || error) && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              transition={{ duration: 0.5 }}
              className="bg-gray-800/50 backdrop-blur-lg rounded-xl p-8 shadow-2xl border border-gray-700"
            >
              <h2 className="text-2xl font-bold mb-6 text-center">Analysis Result</h2>

              {error ? (
                <div className="text-red-400 text-center py-4">
                  <svg className="w-12 h-12 mx-auto mb-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <p className="text-xl">{error}</p>
                </div>
              ) : (
                <motion.div
                  initial={{ rotateY: 90 }}
                  animate={{ rotateY: 0 }}
                  transition={{ duration: 0.6 }}
                  className="flex flex-col items-center"
                >
                  <div className={`w-32 h-32 rounded-full flex items-center justify-center mb-6 shadow-lg ${result === 'Tumor' ? 'bg-red-500/20 border-2 border-red-500' : 'bg-green-500/20 border-2 border-green-500'}`}>
                    {result === 'Tumor' ? (
                      <svg className="w-16 h-16 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                    ) : (
                      <svg className="w-16 h-16 text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                      </svg>
                    )}
                  </div>
                  <h3 className={`text-3xl font-bold mb-2 ${result === 'Tumor' ? 'text-red-400' : 'text-green-400'}`}>
                    {result === 'Tumor' ? 'Tumor Detected' : 'No Tumor Found'}
                  </h3>
                  <p className="text-gray-300 text-center max-w-md">
                    {result === 'Tumor'
                      ? 'Our AI detected signs of a brain tumor. Please consult a medical professional for further evaluation.'
                      : 'No tumor was detected in the scan. For complete certainty, always consult a doctor.'}
                  </p>
                </motion.div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default App;