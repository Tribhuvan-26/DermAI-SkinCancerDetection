import React, { useState, useEffect } from 'react'
import Header from './components/Header.jsx'
import UploadZone from './components/UploadZone.jsx'
import ResultPanel from './components/ResultPanel.jsx'
import ModelStatus from './components/ModelStatus.jsx'
import ClassesGrid from './components/ClassesGrid.jsx'
import Footer from './components/Footer.jsx'
import ParticleBackground from './components/ParticleBackground.jsx'

const API_BASE = 'http://localhost:8000'

export default function App() {
  const [modelStatus, setModelStatus] = useState(null)
  const [uploadedImage, setUploadedImage] = useState(null)   // File object
  const [previewUrl,    setPreviewUrl]    = useState(null)   // data URL
  const [isAnalyzing,  setIsAnalyzing]   = useState(false)
  const [results,      setResults]        = useState(null)
  const [error,        setError]          = useState(null)
  const [activeSection, setActiveSection] = useState('upload')

  // Poll model health on mount
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const res  = await fetch(`${API_BASE}/health`)
        const data = await res.json()
        setModelStatus(data)
      } catch {
        setModelStatus({ status: 'offline', model_loaded: false })
      }
    }
    checkHealth()
    const interval = setInterval(checkHealth, 30_000)
    return () => clearInterval(interval)
  }, [])

  const handleImageSelect = (file) => {
    setUploadedImage(file)
    setResults(null)
    setError(null)
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
  }

  const handleAnalyze = async () => {
    if (!uploadedImage) return
    setIsAnalyzing(true)
    setError(null)
    setResults(null)

    try {
      const formData = new FormData()
      formData.append('file', uploadedImage)

      const res = await fetch(
        `${API_BASE}/predict?topk=7&gradcam=true`,
        { method: 'POST', body: formData }
      )

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Prediction failed')
      }

      const data = await res.json()
      setResults(data)
      setActiveSection('results')
    } catch (err) {
      setError(err.message)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleReset = () => {
    setUploadedImage(null)
    setPreviewUrl(null)
    setResults(null)
    setError(null)
    setActiveSection('upload')
    if (previewUrl) URL.revokeObjectURL(previewUrl)
  }

  return (
    <div className="min-h-screen bg-obsidian-950 bg-grid relative overflow-x-hidden">
      <ParticleBackground />

      {/* Ambient glow orbs */}
      <div className="fixed top-0 left-1/4 w-96 h-96 bg-teal-500/5 rounded-full blur-3xl pointer-events-none" />
      <div className="fixed bottom-1/4 right-1/4 w-64 h-64 bg-teal-600/5 rounded-full blur-3xl pointer-events-none" />

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <Header modelStatus={modelStatus} />

        {/* Model Status Banner */}
        <ModelStatus status={modelStatus} />

        {/* Main Content */}
        <main className="py-8 space-y-8">

          {/* Upload + Result Row */}
          <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
            {/* Upload Panel */}
            <div className={`transition-all duration-500 ${
              activeSection === 'results' ? 'xl:col-span-1' : 'xl:col-span-2'
            }`}>
              <UploadZone
                onImageSelect={handleImageSelect}
                onAnalyze={handleAnalyze}
                onReset={handleReset}
                previewUrl={previewUrl}
                isAnalyzing={isAnalyzing}
                hasResults={!!results}
                modelLoaded={modelStatus?.model_loaded}
              />
            </div>

            {/* Results Panel */}
            {results && (
              <div className="animate-fade-up">
                <ResultPanel
                  results={results}
                  previewUrl={previewUrl}
                />
              </div>
            )}
          </div>

          {/* Error Display */}
          {error && (
            <div className="glass border border-crimson-500/30 rounded-2xl p-5 animate-fade-up">
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-crimson-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="text-crimson-400 text-xs font-bold">!</span>
                </div>
                <div>
                  <p className="text-crimson-400 font-display font-semibold text-sm mb-1">
                    Analysis Failed
                  </p>
                  <p className="text-slate-400 text-sm font-body">{error}</p>
                  {!modelStatus?.model_loaded && (
                    <p className="text-slate-500 text-xs mt-2">
                      Tip: Start the API server with <code className="font-mono text-teal-400 bg-obsidian-800 px-1 rounded">
                        uvicorn api:app --port 8000
                      </code> and ensure a trained model is available.
                    </p>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Classes Reference */}
          <ClassesGrid />
        </main>

        <Footer />
      </div>
    </div>
  )
}
