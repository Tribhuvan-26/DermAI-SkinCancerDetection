import React from 'react'
import { Server, Cpu, Wifi, WifiOff } from 'lucide-react'

export default function ModelStatus({ status }) {
  if (!status) return null

  const isOnline = status.status === 'ok'
  const isLoaded = status.model_loaded

  // Don't show banner if everything is perfect
  if (isOnline && isLoaded) return null

  return (
    <div className={`
      mb-6 rounded-2xl px-5 py-4 border flex items-center gap-4
      animate-fade-up
      ${!isOnline
        ? 'bg-crimson-500/5 border-crimson-500/20'
        : 'bg-amber-500/5 border-amber-500/20'
      }
    `}>
      <div className={`w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0
        ${!isOnline ? 'bg-crimson-500/10' : 'bg-amber-500/10'}`}>
        {!isOnline
          ? <WifiOff size={16} className="text-crimson-400" />
          : <Server  size={16} className="text-amber-400"   />
        }
      </div>

      <div className="flex-1 min-w-0">
        <p className={`font-display font-semibold text-sm
          ${!isOnline ? 'text-crimson-400' : 'text-amber-400'}`}>
          {!isOnline ? 'API Server Offline' : 'Model Not Loaded'}
        </p>
        <p className="text-slate-500 text-xs font-body mt-0.5">
          {!isOnline
            ? 'Start the backend: '
            : 'Run training first, then restart the API: '
          }
          <code className="font-mono text-teal-400 bg-obsidian-800 px-1.5 py-0.5 rounded text-[11px]">
            {!isOnline
              ? 'uvicorn api:app --host 0.0.0.0 --port 8000'
              : 'python train.py --epochs 30'
            }
          </code>
        </p>
      </div>

      <div className="hidden sm:flex items-center gap-3 text-xs font-mono flex-shrink-0">
        <div className="flex items-center gap-1.5 text-slate-600">
          <Wifi size={11} /> localhost:8000
        </div>
        {isOnline && (
          <div className="flex items-center gap-1.5 text-slate-600">
            <Cpu size={11} /> {status.device || 'cpu'}
          </div>
        )}
      </div>
    </div>
  )
}
