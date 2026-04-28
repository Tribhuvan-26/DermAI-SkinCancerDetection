import React from 'react'
import { Activity, Cpu, Github } from 'lucide-react'

export default function Header({ modelStatus }) {
  const isOnline = modelStatus?.status === 'ok'
  const isLoaded = modelStatus?.model_loaded

  return (
    <header className="pt-8 pb-6">
      <div className="flex items-start justify-between gap-4">

        {/* Logo + Brand */}
        <div className="flex items-center gap-4">
          {/* Logo Mark */}
          <div className="relative w-12 h-12 flex-shrink-0">
            <div className="absolute inset-0 rounded-xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-center glow-teal">
              <svg width="26" height="26" viewBox="0 0 26 26" fill="none">
                {/* Stylised DNA/cell helix icon */}
                <circle cx="13" cy="13" r="11" stroke="#2dd4bf" strokeWidth="1.5" strokeDasharray="3 2"/>
                <circle cx="13" cy="13" r="6"  fill="none" stroke="#2dd4bf" strokeWidth="1.5"/>
                <circle cx="13" cy="13" r="2.5" fill="#2dd4bf"/>
                <path d="M7 7 L19 19M19 7 L7 19" stroke="#2dd4bf" strokeWidth="1" opacity="0.4"/>
              </svg>
            </div>
            {/* Pulse ring */}
            <div className="absolute inset-0 rounded-xl border border-teal-500/20 animate-ping opacity-30" />
          </div>

          <div>
            <h1 className="font-display font-bold text-2xl sm:text-3xl tracking-tight text-white leading-none">
              Derm<span className="text-gradient-teal">AI</span>
            </h1>
            <p className="text-slate-500 text-xs font-body mt-1 tracking-wide">
              SKIN CANCER DETECTION · RESNET CNN
            </p>
          </div>
        </div>

        {/* Right controls */}
        <div className="flex items-center gap-3">
          {/* API Status pill */}
          <div className={`
            flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-body font-medium
            border transition-all duration-300
            ${isLoaded
              ? 'bg-emerald-400/5 border-emerald-400/20 text-emerald-400'
              : isOnline
                ? 'bg-amber-400/5 border-amber-400/20 text-amber-400'
                : 'bg-crimson-400/5 border-crimson-400/20 text-crimson-400'
            }
          `}>
            <span className={`w-2 h-2 rounded-full ${
              isLoaded   ? 'bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.8)]' :
              isOnline   ? 'bg-amber-400  shadow-[0_0_6px_rgba(251,191,36,0.8)]'  :
                           'bg-crimson-400 shadow-[0_0_6px_rgba(248,113,113,0.8)]'
            } animate-pulse-slow`} />
            {isLoaded ? 'Model Ready' : isOnline ? 'API Online' : 'API Offline'}
          </div>

          {/* Model badge */}
          {modelStatus?.model_name && (
            <div className="hidden sm:flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-obsidian-700 border border-obsidian-500 text-xs text-slate-400 font-mono">
              <Cpu size={11} className="text-teal-400" />
              {modelStatus.model_name.toUpperCase()}
            </div>
          )}

          {/* GitHub link */}
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="w-9 h-9 rounded-xl glass flex items-center justify-center text-slate-500 hover:text-teal-400 hover:border-teal-500/30 transition-all duration-200"
          >
            <Github size={16} />
          </a>
        </div>
      </div>

      {/* Subtitle strip */}
      <div className="mt-6 flex items-center gap-3">
        <div className="h-px flex-1 bg-gradient-to-r from-teal-500/20 to-transparent" />
        <p className="text-slate-500 text-xs font-body tracking-widest uppercase">
          AI-Powered Dermatological Analysis
        </p>
        <div className="h-px flex-1 bg-gradient-to-l from-teal-500/20 to-transparent" />
      </div>
    </header>
  )
}
