import React, { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, ImageIcon, X, Zap, RotateCcw, AlertCircle } from 'lucide-react'

const ACCEPTED_TYPES = { 'image/jpeg': ['.jpg', '.jpeg'], 'image/png': ['.png'] }
const MAX_SIZE_MB    = 10

export default function UploadZone({
  onImageSelect, onAnalyze, onReset,
  previewUrl, isAnalyzing, hasResults, modelLoaded
}) {
  const [dragError, setDragError] = useState(null)

  const onDrop = useCallback((accepted, rejected) => {
    setDragError(null)
    if (rejected.length > 0) {
      const err = rejected[0].errors[0]
      if (err.code === 'file-too-large')
        setDragError(`File too large. Max size: ${MAX_SIZE_MB}MB`)
      else if (err.code === 'file-invalid-type')
        setDragError('Only JPEG and PNG images are accepted.')
      else
        setDragError(err.message)
      return
    }
    if (accepted.length > 0) onImageSelect(accepted[0])
  }, [onImageSelect])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept:   ACCEPTED_TYPES,
    maxSize:  MAX_SIZE_MB * 1024 * 1024,
    maxFiles: 1,
    disabled: isAnalyzing,
  })

  return (
    <div className="glass-bright rounded-3xl overflow-hidden">
      {/* Card header */}
      <div className="px-6 pt-6 pb-4 border-b border-white/5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-teal-500/10 border border-teal-500/20 flex items-center justify-center">
              <ImageIcon size={15} className="text-teal-400" />
            </div>
            <div>
              <h2 className="font-display font-semibold text-white text-sm">
                Image Analysis
              </h2>
              <p className="text-slate-500 text-xs font-body">Upload a skin lesion image</p>
            </div>
          </div>

          {previewUrl && (
            <button
              onClick={onReset}
              disabled={isAnalyzing}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs text-slate-400 hover:text-white border border-white/5 hover:border-white/15 transition-all duration-200 disabled:opacity-50"
            >
              <RotateCcw size={12} />
              Reset
            </button>
          )}
        </div>
      </div>

      <div className="p-6 space-y-5">

        {/* Dropzone OR Preview */}
        {!previewUrl ? (
          <div
            {...getRootProps()}
            className={`
              relative rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer
              min-h-[260px] flex flex-col items-center justify-center gap-4 p-8 text-center
              ${isDragActive && !isDragReject
                ? 'border-teal-400/60 bg-teal-500/5 scale-[1.01]'
                : isDragReject
                  ? 'border-crimson-400/60 bg-crimson-500/5'
                  : 'border-obsidian-500 hover:border-teal-500/40 hover:bg-teal-500/3'
              }
            `}
          >
            <input {...getInputProps()} />

            {/* Animated upload icon */}
            <div className={`
              w-16 h-16 rounded-2xl flex items-center justify-center transition-all duration-300
              ${isDragActive
                ? 'bg-teal-500/20 scale-110'
                : 'bg-obsidian-600 group-hover:bg-teal-500/10'
              }
            `}>
              <Upload
                size={28}
                className={`transition-colors duration-300 ${
                  isDragReject ? 'text-crimson-400' :
                  isDragActive ? 'text-teal-400'    : 'text-slate-500'
                }`}
              />
            </div>

            <div>
              <p className="text-white font-display font-semibold text-base mb-1">
                {isDragActive
                  ? isDragReject ? 'Invalid file type' : 'Drop image here'
                  : 'Drop your image here'
                }
              </p>
              <p className="text-slate-500 text-sm font-body">
                or <span className="text-teal-400 font-medium">click to browse</span>
              </p>
            </div>

            <div className="flex items-center gap-4 text-xs text-slate-600">
              <span>JPEG · PNG</span>
              <span className="w-1 h-1 rounded-full bg-slate-600" />
              <span>Max {MAX_SIZE_MB}MB</span>
              <span className="w-1 h-1 rounded-full bg-slate-600" />
              <span>224×224+ recommended</span>
            </div>

            {/* Corner accents */}
            <div className="absolute top-3 left-3 w-4 h-4 border-t border-l border-teal-500/20 rounded-tl" />
            <div className="absolute top-3 right-3 w-4 h-4 border-t border-r border-teal-500/20 rounded-tr" />
            <div className="absolute bottom-3 left-3 w-4 h-4 border-b border-l border-teal-500/20 rounded-bl" />
            <div className="absolute bottom-3 right-3 w-4 h-4 border-b border-r border-teal-500/20 rounded-br" />
          </div>
        ) : (
          /* Image Preview */
          <div className="relative rounded-2xl overflow-hidden bg-obsidian-800 group">
            <img
              src={previewUrl}
              alt="Uploaded skin lesion"
              className="w-full h-64 object-cover"
            />

            {/* Scanning animation overlay */}
            {isAnalyzing && (
              <div className="absolute inset-0 flex flex-col">
                {/* Scan line */}
                <div className="absolute w-full h-0.5 bg-gradient-to-r from-transparent via-teal-400 to-transparent animate-scan opacity-80" />
                {/* Dark overlay */}
                <div className="absolute inset-0 bg-obsidian-950/40" />
                {/* Grid overlay */}
                <div className="absolute inset-0 bg-grid opacity-40" />
                {/* Processing label */}
                <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex items-center gap-2 glass px-4 py-2 rounded-full">
                  <div className="w-2 h-2 rounded-full bg-teal-400 animate-pulse" />
                  <span className="text-teal-400 text-xs font-mono font-medium tracking-wider">
                    ANALYZING…
                  </span>
                </div>
              </div>
            )}

            {/* Corner badge: filename */}
            <div className="absolute top-3 left-3 glass px-2.5 py-1 rounded-lg text-xs text-slate-300 font-mono max-w-[180px] truncate">
              {previewUrl ? 'image.jpg' : ''}
            </div>
          </div>
        )}

        {/* Drag error */}
        {dragError && (
          <div className="flex items-center gap-2 text-crimson-400 text-sm font-body">
            <AlertCircle size={14} />
            {dragError}
          </div>
        )}

        {/* Action Buttons */}
        {previewUrl && (
          <div className="flex gap-3">
            <button
              onClick={onAnalyze}
              disabled={isAnalyzing || !modelLoaded}
              className={`
                flex-1 flex items-center justify-center gap-2.5 py-3.5 rounded-xl
                font-display font-semibold text-sm tracking-wide
                transition-all duration-300
                ${isAnalyzing
                  ? 'bg-teal-500/20 text-teal-400 cursor-wait border border-teal-500/30'
                  : !modelLoaded
                    ? 'bg-obsidian-600 text-slate-500 cursor-not-allowed border border-obsidian-500'
                    : 'bg-teal-500 hover:bg-teal-400 text-obsidian-950 glow-teal hover:glow-teal-strong border border-teal-400/50 hover:scale-[1.02] active:scale-[0.98]'
                }
              `}
            >
              {isAnalyzing ? (
                <>
                  <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"/>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                  </svg>
                  Analyzing Image…
                </>
              ) : (
                <>
                  <Zap size={16} />
                  {!modelLoaded ? 'Model Not Ready' : 'Analyze Image'}
                </>
              )}
            </button>

            {!isAnalyzing && (
              <button
                {...getRootProps()}
                className="px-4 py-3.5 rounded-xl glass border border-white/10 hover:border-teal-500/30 text-slate-400 hover:text-white transition-all duration-200"
                title="Upload different image"
              >
                <input {...getInputProps()} />
                <Upload size={16} />
              </button>
            )}
          </div>
        )}

        {/* Tips */}
        {!previewUrl && (
          <div className="grid grid-cols-3 gap-3">
            {[
              { icon: '🔬', label: 'High-res', desc: 'Use clear, well-lit images' },
              { icon: '📐', label: 'Centered', desc: 'Lesion should fill the frame' },
              { icon: '🎨', label: 'Color', desc: 'Avoid filters or overlays' },
            ].map((tip) => (
              <div key={tip.label} className="bg-obsidian-800/50 rounded-xl p-3 border border-white/5">
                <div className="text-lg mb-1">{tip.icon}</div>
                <p className="text-white text-xs font-display font-semibold">{tip.label}</p>
                <p className="text-slate-500 text-xs font-body mt-0.5">{tip.desc}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
