import React from 'react'
import { Heart, Github, ExternalLink } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="py-8 mt-4">
      <div className="h-px bg-gradient-to-r from-transparent via-teal-500/20 to-transparent mb-6" />

      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-xs text-slate-600 font-body">
        <div className="flex items-center gap-1.5">
          <span>Built with</span>
          <Heart size={11} className="text-crimson-500 fill-crimson-500" />
          <span>using <span className="text-slate-500 font-mono">PyTorch · ResNet · FastAPI · React</span></span>
        </div>

        <div className="flex items-center gap-4">
          <a href="https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection"
             target="_blank" rel="noopener noreferrer"
             className="flex items-center gap-1 hover:text-teal-400 transition-colors">
            HAM10000 Dataset <ExternalLink size={10} />
          </a>
          <a href="https://github.com" target="_blank" rel="noopener noreferrer"
             className="flex items-center gap-1 hover:text-teal-400 transition-colors">
            <Github size={11} /> Source Code
          </a>
        </div>

        <div className="text-center sm:text-right">
          <p className="text-slate-700">For research purposes only. Not medical advice.</p>
          <p className="text-slate-700 mt-0.5 font-mono">DermAI v1.0.0 · ResNet CNN</p>
        </div>
      </div>
    </footer>
  )
}
