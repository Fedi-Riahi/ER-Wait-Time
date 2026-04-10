import { useState, useEffect } from 'react'
import { fetchRuns } from '../services/api'
import { ChevronDown, ChevronUp, RefreshCw } from 'lucide-react'

export default function Experiments() {
  const [runs,     setRuns]     = useState([])
  const [loading,  setLoading]  = useState(true)
  const [expanded, setExpanded] = useState(null)

  const load = () => {
    setLoading(true)
    fetchRuns()
      .then(setRuns)
      .catch(() => setRuns([]))
      .finally(() => setLoading(false))
  }

  useEffect(() => { load() }, [])

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-xl font-bold text-white">MLflow Experiments</h1>
          <p className="text-sm text-gray-500">{runs.length} runs tracked</p>
        </div>
        <button onClick={load}
          className="flex items-center gap-2 px-3 py-2 rounded-lg bg-dark-800 border border-dark-600 text-gray-400 hover:text-white text-sm">
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {loading ? (
        <div className="text-center text-gray-500 py-20">Loading MLflow runs...</div>
      ) : runs.length === 0 ? (
        <div className="text-center text-gray-500 py-20">
          No runs found. Run <code className="text-accent">train.py</code> first.
        </div>
      ) : (
        runs.map(r => (
          <div key={r.id} className="bg-dark-800 border border-dark-600 rounded-xl overflow-hidden">
            <div
              className="flex items-center gap-4 px-5 py-4 cursor-pointer hover:bg-dark-700 transition-colors"
              onClick={() => setExpanded(expanded === r.id ? null : r.id)}
            >
              <div className="font-mono text-xs text-gray-500 w-20 shrink-0">{r.id}</div>
              <div className="flex-1 text-sm font-semibold text-white">{r.name || r.model}</div>
              <div className="flex gap-6 text-sm">
                <span className="text-red-400">RMSE <b>{r.metrics.rmse}</b></span>
                <span className="text-yellow-400">MAE <b>{r.metrics.mae}</b></span>
                <span className="text-blue-400">R² <b>{r.metrics.r2}</b></span>
              </div>
              <span className={`px-2 py-1 rounded text-xs font-semibold
                ${r.status==='best' ? 'bg-emerald-500/20 text-emerald-400' : 'bg-gray-500/20 text-gray-400'}`}>
                {r.status==='best' ? '⭐ Best' : 'Done'}
              </span>
              {expanded===r.id ? <ChevronUp size={14} className="text-gray-500"/> : <ChevronDown size={14} className="text-gray-500"/>}
            </div>

            {expanded === r.id && (
              <div className="border-t border-dark-600 px-5 py-4 bg-dark-900">
                <div className="grid grid-cols-4 gap-3 mb-4">
                  {Object.entries(r.metrics).map(([k,v]) => (
                    <div key={k} className="bg-dark-700 rounded-lg p-3 text-center">
                      <div className="text-xs text-gray-500 uppercase mb-1">{k}</div>
                      <div className="font-bold text-white">{v}</div>
                    </div>
                  ))}
                </div>
                <div className="text-xs text-gray-400 mb-2 font-semibold">Parameters</div>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(r.params).map(([k,v]) => (
                    <span key={k} className="bg-dark-700 border border-dark-600 rounded px-3 py-1 text-xs font-mono">
                      {k}: <span className="text-accent">{v}</span>
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        ))
      )}
    </div>
  )
}
