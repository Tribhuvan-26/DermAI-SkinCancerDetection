import React, { useState } from 'react'
import { RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer,
         BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts'
import { ShieldAlert, ShieldCheck, AlertTriangle, Clock, Brain, ChevronDown, ChevronUp, Eye } from 'lucide-react'

const CLASS_FULL_NAMES = {
  nv:    'Melanocytic Nevi',
  mel:   'Melanoma',
  bkl:   'Benign Keratosis',
  bcc:   'Basal Cell Carcinoma',
  akiec: 'Actinic Keratoses',
  vasc:  'Vascular Lesions',
  df:    'Dermatofibroma',
}

const RISK_CONFIG = {
  high:    { color: '#ef4444', bg: 'bg-crimson-500/10', border: 'border-crimson-500/30', icon: ShieldAlert,   label: 'HIGH RISK',   bar: '#ef4444' },
  medium:  { color: '#f59e0b', bg: 'bg-amber-500/10',   border: 'border-amber-500/30',   icon: AlertTriangle, label: 'MODERATE',    bar: '#f59e0b' },
  low:     { color: '#10b981', bg: 'bg-emerald-500/10', border: 'border-emerald-500/30', icon: ShieldCheck,   label: 'LOW RISK',    bar: '#10b981' },
  unknown: { color: '#94a3b8', bg: 'bg-slate-500/10',   border: 'border-slate-500/30',   icon: ShieldCheck,   label: 'UNKNOWN',     bar: '#94a3b8' },
}

const ADVICE = {
  high:    'Seek immediate dermatological consultation. Do not delay.',
  medium:  'Schedule a dermatologist appointment within 2–4 weeks.',
  low:     'Monitor for changes. Annual skin checks are recommended.',
  unknown: 'Consult a healthcare professional for proper evaluation.',
}

// Custom tooltip for bar chart
const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass px-3 py-2 rounded-lg text-xs border border-white/10">
        <p className="text-white font-display font-semibold">{payload[0].payload.name}</p>
        <p className="text-teal-400 font-mono">{(payload[0].value * 100).toFixed(2)}%</p>
      </div>
    )
  }
  return null
}

// Animated confidence bar
function ConfidenceBar({ value, color, delay = 0 }) {
  return (
    <div className="h-1.5 rounded-full bg-obsidian-600 overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-700"
        style={{
          width: `${value * 100}%`,
          background: color,
          transitionDelay: `${delay}ms`,
          boxShadow: `0 0 8px ${color}60`,
        }}
      />
    </div>
  )
}

export default function ResultPanel({ results, previewUrl }) {
  const [showGradcam, setShowGradcam] = useState(false)
  const [expandAll,   setExpandAll]   = useState(false)

  if (!results) return null

  const { predictions, top_prediction, gradcam_base64, inference_ms, model_name } = results
  const risk    = RISK_CONFIG[top_prediction.risk] || RISK_CONFIG.unknown
  const RiskIcon = risk.icon

  // Format predictions for charts
  const chartData = predictions.map(p => ({
    name:  p.class_code.toUpperCase(),
    full:  CLASS_FULL_NAMES[p.class_code] || p.class_code,
    value: p.confidence,
    risk:  p.risk,
    color: RISK_CONFIG[p.risk]?.bar || '#94a3b8',
  }))

  const radarData = predictions.map(p => ({
    subject: p.class_code.toUpperCase(),
    value:   Math.round(p.confidence * 100),
    fullMark:100,
  }))

  const visiblePredictions = expandAll ? predictions : predictions.slice(0, 4)

  return (
    <div className="glass-bright rounded-3xl overflow-hidden h-full">
      {/* Header */}
      <div className="px-6 pt-6 pb-4 border-b border-white/5">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-teal-500/10 border border-teal-500/20 flex items-center justify-center">
              <Brain size={15} className="text-teal-400" />
            </div>
            <div>
              <h2 className="font-display font-semibold text-white text-sm">Analysis Results</h2>
              <p className="text-slate-500 text-xs font-body">{model_name?.toUpperCase()} · {inference_ms}ms</p>
            </div>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-slate-500 font-mono">
            <Clock size={11} />
            {inference_ms}ms
          </div>
        </div>
      </div>

      <div className="p-6 space-y-5 overflow-y-auto max-h-[calc(100vh-200px)]">

        {/* ── Primary Diagnosis Card ─────────────────────────────────── */}
        <div className={`relative rounded-2xl p-5 border ${risk.bg} ${risk.border} overflow-hidden`}>
          {/* Glow accent */}
          <div className="absolute top-0 right-0 w-24 h-24 rounded-full blur-2xl opacity-20"
               style={{ background: risk.color }} />

          <div className="relative">
            <div className="flex items-start justify-between gap-3 mb-3">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-mono tracking-widest"
                        style={{ color: risk.color }}>
                    {risk.label}
                  </span>
                  <RiskIcon size={13} style={{ color: risk.color }} />
                </div>
                <h3 className="font-display font-bold text-white text-xl leading-tight">
                  {top_prediction.class_name}
                </h3>
                <p className="text-slate-400 text-xs font-mono mt-0.5">
                  CODE: {top_prediction.class_code.toUpperCase()}
                </p>
              </div>

              {/* Big confidence circle */}
              <div className="flex-shrink-0 relative w-16 h-16">
                <svg viewBox="0 0 56 56" className="w-full h-full -rotate-90">
                  <circle cx="28" cy="28" r="22" fill="none"
                          stroke="rgba(255,255,255,0.05)" strokeWidth="4"/>
                  <circle cx="28" cy="28" r="22" fill="none"
                          stroke={risk.color} strokeWidth="4"
                          strokeLinecap="round"
                          strokeDasharray={`${2 * Math.PI * 22}`}
                          strokeDashoffset={`${2 * Math.PI * 22 * (1 - top_prediction.confidence)}`}
                          style={{ transition: 'stroke-dashoffset 1s ease' }}
                  />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center">
                  <span className="font-display font-bold text-white text-sm leading-none">
                    {Math.round(top_prediction.confidence * 100)}%
                  </span>
                  <span className="text-slate-500 text-[9px] font-mono leading-none mt-0.5">CONF</span>
                </div>
              </div>
            </div>

            <p className="text-slate-400 text-sm font-body leading-relaxed mb-3">
              {top_prediction.description}
            </p>

            {/* Clinical advice */}
            <div className="flex items-start gap-2 rounded-xl bg-black/20 px-3 py-2.5">
              <span className="text-base flex-shrink-0 mt-0.5">💡</span>
              <p className="text-slate-300 text-xs font-body leading-relaxed">
                <strong className="text-white">Clinical advice:</strong>{' '}
                {ADVICE[top_prediction.risk]}
              </p>
            </div>
          </div>
        </div>

        {/* ── Grad-CAM Section ────────────────────────────────────────── */}
        {gradcam_base64 && (
          <div className="rounded-2xl overflow-hidden border border-white/5 bg-obsidian-800/50">
            <button
              onClick={() => setShowGradcam(!showGradcam)}
              className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/3 transition-colors"
            >
              <div className="flex items-center gap-2 text-sm font-display font-semibold text-white">
                <Eye size={15} className="text-teal-400" />
                Grad-CAM Explainability
              </div>
              {showGradcam ? <ChevronUp size={15} className="text-slate-500" />
                           : <ChevronDown size={15} className="text-slate-500" />}
            </button>

            {showGradcam && (
              <div className="px-4 pb-4 space-y-3 animate-fade-in">
                <p className="text-slate-500 text-xs font-body">
                  Heat map shows which regions influenced the prediction most.
                  <span className="text-red-400"> Red</span> = high importance,
                  <span className="text-blue-400"> Blue</span> = low importance.
                </p>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <p className="text-slate-600 text-xs font-mono mb-1.5">ORIGINAL</p>
                    <img src={previewUrl} alt="Original"
                         className="w-full rounded-xl object-cover aspect-square" />
                  </div>
                  <div>
                    <p className="text-slate-600 text-xs font-mono mb-1.5">GRAD-CAM</p>
                    <img src={`data:image/png;base64,${gradcam_base64}`}
                         alt="Grad-CAM overlay"
                         className="w-full rounded-xl object-cover aspect-square" />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── All Predictions List ─────────────────────────────────────── */}
        <div>
          <div className="flex items-center justify-between mb-3">
            <h4 className="text-slate-400 text-xs font-mono tracking-widest uppercase">
              All Probabilities
            </h4>
            <button
              onClick={() => setExpandAll(!expandAll)}
              className="text-xs text-teal-400 hover:text-teal-300 font-body transition-colors"
            >
              {expandAll ? 'Show less' : `Show all ${predictions.length}`}
            </button>
          </div>

          <div className="space-y-2.5">
            {visiblePredictions.map((p, i) => {
              const rc = RISK_CONFIG[p.risk] || RISK_CONFIG.unknown
              const isTop = i === 0
              return (
                <div
                  key={p.class_code}
                  className={`rounded-xl p-3 border transition-all duration-200
                    ${isTop
                      ? `${rc.bg} ${rc.border}`
                      : 'bg-obsidian-800/40 border-white/5'
                    }`}
                  style={{ animationDelay: `${i * 60}ms` }}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <div className="flex items-center gap-2">
                      {isTop && (
                        <span className="text-[10px] font-mono px-1.5 py-0.5 rounded"
                              style={{ background: `${rc.color}20`, color: rc.color }}>
                          TOP
                        </span>
                      )}
                      <span className="text-white text-sm font-body font-medium">
                        {p.class_name}
                      </span>
                    </div>
                    <span className="font-mono text-sm font-semibold"
                          style={{ color: isTop ? rc.color : '#94a3b8' }}>
                      {(p.confidence * 100).toFixed(2)}%
                    </span>
                  </div>
                  <ConfidenceBar
                    value={p.confidence}
                    color={rc.bar}
                    delay={i * 80}
                  />
                </div>
              )
            })}
          </div>
        </div>

        {/* ── Bar Chart ────────────────────────────────────────────────── */}
        <div className="rounded-2xl bg-obsidian-800/50 border border-white/5 p-4">
          <h4 className="text-slate-400 text-xs font-mono tracking-widest uppercase mb-4">
            Confidence Distribution
          </h4>
          <ResponsiveContainer width="100%" height={160}>
            <BarChart data={chartData} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
              <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} tickFormatter={v => `${Math.round(v*100)}%`} />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
              <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                {chartData.map((entry, idx) => (
                  <Cell key={idx} fill={entry.color} opacity={idx === 0 ? 1 : 0.6} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* ── Radar Chart ──────────────────────────────────────────────── */}
        <div className="rounded-2xl bg-obsidian-800/50 border border-white/5 p-4">
          <h4 className="text-slate-400 text-xs font-mono tracking-widest uppercase mb-4">
            Probability Radar
          </h4>
          <ResponsiveContainer width="100%" height={200}>
            <RadarChart data={radarData} margin={{ top: 10, right: 20, bottom: 10, left: 20 }}>
              <PolarGrid stroke="rgba(255,255,255,0.05)" />
              <PolarAngleAxis
                dataKey="subject"
                tick={{ fill: '#64748b', fontSize: 10, fontFamily: 'JetBrains Mono' }}
              />
              <Radar
                name="Confidence"
                dataKey="value"
                stroke="#2dd4bf"
                fill="#2dd4bf"
                fillOpacity={0.15}
                strokeWidth={2}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        {/* ── Disclaimer ───────────────────────────────────────────────── */}
        <div className="rounded-xl bg-obsidian-800/30 border border-white/5 p-3">
          <p className="text-slate-600 text-[11px] font-body leading-relaxed text-center">
            ⚠️ This tool is for <strong className="text-slate-500">research purposes only</strong> and does not constitute medical advice.
            Always consult a qualified dermatologist for diagnosis and treatment.
          </p>
        </div>
      </div>
    </div>
  )
}
