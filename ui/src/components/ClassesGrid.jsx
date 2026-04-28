import React, { useState } from 'react'
import { BookOpen, ChevronDown, ChevronUp } from 'lucide-react'

const CLASSES = [
  {
    code: 'nv',    abbr: 'NV',
    name: 'Melanocytic Nevi',
    risk: 'low',
    prevalence: '67%',
    description: 'Common benign moles formed from melanocytes. Usually harmless but should be monitored for asymmetry, border irregularity, color variation, or diameter changes (ABCD rule).',
    symptoms: ['Round/oval shape', 'Uniform color', 'Well-defined borders', '<6mm diameter'],
    emoji: '🟤',
  },
  {
    code: 'mel',   abbr: 'MEL',
    name: 'Melanoma',
    risk: 'high',
    prevalence: '11%',
    description: 'The most dangerous form of skin cancer arising from melanocytes. Early detection is critical — 5-year survival rate drops from 98% (local) to 23% (distant metastasis).',
    symptoms: ['Asymmetric shape', 'Irregular border', 'Multiple colors', 'Diameter >6mm'],
    emoji: '⚫',
  },
  {
    code: 'bkl',   abbr: 'BKL',
    name: 'Benign Keratosis',
    risk: 'low',
    prevalence: '11%',
    description: 'Non-cancerous skin growths including seborrheic keratoses, solar lentigines, and lichen planus-like keratoses. Very common in older adults and generally harmless.',
    symptoms: ['Waxy/scaly texture', 'Stuck-on appearance', 'Light to dark brown', 'Flat or slightly raised'],
    emoji: '🟫',
  },
  {
    code: 'bcc',   abbr: 'BCC',
    name: 'Basal Cell Carcinoma',
    risk: 'medium',
    prevalence: '5%',
    description: 'The most common form of skin cancer. Rarely metastasizes but can cause significant local tissue damage if untreated. Highly treatable when caught early.',
    symptoms: ['Pearly/waxy bump', 'Pink skin growth', 'Scar-like lesion', 'Bleeding ulcer'],
    emoji: '🔴',
  },
  {
    code: 'akiec', abbr: 'AKIEC',
    name: 'Actinic Keratoses',
    risk: 'medium',
    prevalence: '3%',
    description: 'Pre-cancerous rough scaly patches caused by years of UV exposure. Approximately 10% may progress to squamous cell carcinoma if untreated.',
    symptoms: ['Rough/scaly patch', 'Flat to slightly raised', 'Pink, red, or brown', 'Itching or burning'],
    emoji: '🟡',
  },
  {
    code: 'vasc',  abbr: 'VASC',
    name: 'Vascular Lesions',
    risk: 'low',
    prevalence: '1%',
    description: 'Benign vascular tumors and malformations including cherry angiomas, angiokeratomas, and pyogenic granulomas. Usually require no treatment unless cosmetically bothersome.',
    symptoms: ['Bright red color', 'Dome-shaped', 'Soft/compressible', 'Bleed easily'],
    emoji: '🔵',
  },
  {
    code: 'df',    abbr: 'DF',
    name: 'Dermatofibroma',
    risk: 'low',
    prevalence: '1%',
    description: 'Benign fibrous nodules in the skin, most commonly found on the legs. They feel firm and may dimple inward when pinched. Very rarely malignant transformation occurs.',
    symptoms: ['Firm nodule', 'Dimple sign', 'Pink to brown', 'Typically <1cm'],
    emoji: '⬤',
  },
]

const RISK_STYLE = {
  high:   { pill: 'bg-crimson-500/15 text-crimson-400 border-crimson-500/30',  dot: 'bg-crimson-400',  label: 'High Risk'  },
  medium: { pill: 'bg-amber-500/15  text-amber-400  border-amber-500/30',      dot: 'bg-amber-400',    label: 'Moderate'   },
  low:    { pill: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30',  dot: 'bg-emerald-400',  label: 'Low Risk'   },
}

function ClassCard({ cls }) {
  const [expanded, setExpanded] = useState(false)
  const rs = RISK_STYLE[cls.risk]

  return (
    <div className={`
      glass rounded-2xl overflow-hidden border transition-all duration-300
      hover:border-teal-500/20 hover:glow-teal group
      ${expanded ? 'border-teal-500/15' : 'border-white/5'}
    `}>
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full text-left p-4"
      >
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-obsidian-700 flex items-center justify-center text-xl flex-shrink-0 border border-white/5">
              {cls.emoji}
            </div>
            <div className="min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-mono text-teal-400 text-xs bg-teal-500/10 px-1.5 py-0.5 rounded border border-teal-500/20">
                  {cls.abbr}
                </span>
                <span className={`text-[10px] font-mono px-1.5 py-0.5 rounded border ${rs.pill}`}>
                  {rs.label}
                </span>
              </div>
              <h3 className="font-display font-semibold text-white text-sm mt-1 leading-tight">
                {cls.name}
              </h3>
            </div>
          </div>

          <div className="flex items-center gap-2 flex-shrink-0">
            <span className="text-xs text-slate-500 font-mono hidden sm:block">
              {cls.prevalence}
            </span>
            {expanded
              ? <ChevronUp size={14} className="text-teal-400" />
              : <ChevronDown size={14} className="text-slate-500 group-hover:text-teal-400 transition-colors" />
            }
          </div>
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-3 border-t border-white/5 pt-3 animate-fade-in">
          <p className="text-slate-400 text-sm font-body leading-relaxed">
            {cls.description}
          </p>

          <div>
            <p className="text-slate-600 text-xs font-mono uppercase tracking-wider mb-2">
              Key Features
            </p>
            <div className="grid grid-cols-2 gap-1.5">
              {cls.symptoms.map((s, i) => (
                <div key={i} className="flex items-center gap-1.5 text-xs text-slate-400 font-body">
                  <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${rs.dot}`} />
                  {s}
                </div>
              ))}
            </div>
          </div>

          <div className="flex items-center justify-between text-xs text-slate-600">
            <span className="font-mono">Dataset prevalence: <span className="text-slate-400">{cls.prevalence}</span></span>
            <span className={`flex items-center gap-1 ${
              cls.risk === 'high' ? 'text-crimson-400' :
              cls.risk === 'medium' ? 'text-amber-400' : 'text-emerald-400'
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${rs.dot} animate-pulse-slow`} />
              {rs.label}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

export default function ClassesGrid() {
  const [visible, setVisible] = useState(false)

  return (
    <div className="glass-bright rounded-3xl overflow-hidden">
      <button
        onClick={() => setVisible(!visible)}
        className="w-full px-6 py-5 flex items-center justify-between hover:bg-white/3 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-teal-500/10 border border-teal-500/20 flex items-center justify-center">
            <BookOpen size={15} className="text-teal-400" />
          </div>
          <div className="text-left">
            <h2 className="font-display font-semibold text-white text-sm">
              Disease Reference Guide
            </h2>
            <p className="text-slate-500 text-xs font-body">
              7 HAM10000 lesion classes with clinical details
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="hidden sm:flex gap-1.5">
            {['high', 'medium', 'low'].map(r => (
              <span key={r} className={`text-[10px] font-mono px-2 py-0.5 rounded border ${RISK_STYLE[r].pill}`}>
                {RISK_STYLE[r].label}
              </span>
            ))}
          </div>
          {visible
            ? <ChevronUp size={16} className="text-teal-400 flex-shrink-0" />
            : <ChevronDown size={16} className="text-slate-500 flex-shrink-0" />
          }
        </div>
      </button>

      {visible && (
        <div className="px-6 pb-6 border-t border-white/5">
          <div className="pt-5 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3 animate-fade-in">
            {CLASSES.map(cls => (
              <ClassCard key={cls.code} cls={cls} />
            ))}
          </div>

          <div className="mt-4 pt-4 border-t border-white/5 flex items-center justify-between text-xs text-slate-600">
            <span>Source: HAM10000 — Human Against Machine with 10,000 training images</span>
            <a href="https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection"
               target="_blank" rel="noopener noreferrer"
               className="text-teal-400 hover:underline font-mono">
              Kaggle Dataset →
            </a>
          </div>
        </div>
      )}
    </div>
  )
}
