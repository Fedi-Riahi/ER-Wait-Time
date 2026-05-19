import { useState, useEffect } from 'react'
import { fetchModels, trainModel } from '../services/api'
import toast from 'react-hot-toast'

export default function Training() {
  const [models,     setModels]     = useState([])
  const [selected,   setSelected]   = useState(null)   // key string
  const [params,     setParams]     = useState({})      // { [paramKey]: value }
  const [usePCA,     setUsePCA]     = useState(false)
  const [isTraining, setIsTraining] = useState(false)
  const [progress,   setProgress]   = useState(0)
  const [results,    setResults]    = useState(null)
  const [loading,    setLoading]    = useState(true)

  // ── Fetch model list + their hyperparam schemas from backend ──────────────
  useEffect(() => {
    fetchModels()
      .then(data => {
        setModels(data)
        if (data.length) pickModel(data[0], data)
      })
      .catch(() => toast.error('Could not load models — is the backend running?'))
      .finally(() => setLoading(false))
  }, [])

  // ── Select a model and seed params with its defaults ──────────────────────
  const pickModel = (model, modelList = models) => {
    const full   = modelList.find(m => m.key === model.key) ?? model
    const schema = full.hyperparams ?? []
    const defaults = {}
    schema.forEach(p => { defaults[p.key] = p.default })
    setSelected(full.key)
    setParams(defaults)
    setResults(null)
  }

  const setParam = (k, v) => setParams(p => ({ ...p, [k]: v }))

  // ── Train ──────────────────────────────────────────────────────────────────
  const train = async () => {
    if (!selected) return
    setIsTraining(true)
    setProgress(0)
    setResults(null)

    const iv = setInterval(() => {
      setProgress(p => { if (p >= 90) { clearInterval(iv); return 90 } return p + 10 })
    }, 400)

    const id = toast.loading(`Training ${activeModel?.name ?? selected}…`)
    try {
      const finalParams = { ...params }

      // neural_net: convert tuple-string → JS array for the backend
      if (selected === 'neural_net' && typeof finalParams.hidden_layer_sizes === 'string') {
        finalParams.hidden_layer_sizes = JSON.parse(
          finalParams.hidden_layer_sizes.replace(/\(/g, '[').replace(/\)/g, ']')
        )
      }

      // random_forest / any model: bootstrap select string → real boolean
      if ('bootstrap' in finalParams) {
        finalParams.bootstrap =
          finalParams.bootstrap === true || finalParams.bootstrap === 'true'
      }

      const res = await trainModel({ model: selected, hyperparams: finalParams, use_pca: usePCA })
      setResults(res)
      setProgress(100)
      toast.success('Training complete!', { id })
    } catch {
      toast.error('Backend offline — start uvicorn first', { id })
    } finally {
      clearInterval(iv)
      setIsTraining(false)
    }
  }

  const activeModel = models.find(m => m.key === selected)
  const hparams     = activeModel?.hyperparams ?? []

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="grid grid-cols-3 gap-5">

      {/* ── 1. Model selector ──────────────────────────────────────────────── */}
      <div className="space-y-4">
        <div className="bg-dark-800 border border-dark-600 rounded-xl p-4">
          <div className="text-sm font-semibold text-gray-300 mb-3">Select Model</div>

          {loading ? (
            <div className="text-xs text-gray-500 text-center py-8 animate-pulse">
              Loading models…
            </div>
          ) : (
            <div className="space-y-2">
              {models.map(m => (
                <div
                  key={m.key}
                  onClick={() => pickModel(m)}
                  className={`cursor-pointer rounded-lg border p-3 transition-all
                    ${selected === m.key
                      ? 'border-accent/60 bg-accent/10'
                      : 'border-dark-600 hover:border-dark-500'}`}
                >
                  <div className="text-sm font-semibold text-white">{m.name}</div>
                  <div className="text-xs text-gray-500">{m.family}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* ── 2. Hyperparameters ─────────────────────────────────────────────── */}
      <div className="space-y-4">
        <div className="bg-dark-800 border border-dark-600 rounded-xl p-4">
          <div className="text-sm font-semibold text-gray-300 mb-4">Hyperparameters</div>

          {hparams.length === 0 ? (
            <div className="text-xs text-gray-500 text-center py-8">
              {loading ? 'Loading…' : 'Select a model to configure'}
            </div>
          ) : (
            <div className="space-y-4">
              {hparams.map(p => (
                <div key={p.key}>
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-300">{p.label}</span>
                    <span className="text-accent font-mono">
                      {params[p.key] ?? p.default}
                    </span>
                  </div>

                  {p.type === 'range' ? (
                    <input
                      type="range"
                      min={p.min}
                      max={p.max}
                      step={p.step}
                      value={params[p.key] ?? p.default}
                      onChange={e => setParam(p.key, parseFloat(e.target.value))}
                      className="w-full accent-blue-500 cursor-pointer"
                    />
                  ) : (
                    <select
                      value={params[p.key] ?? p.default}
                      onChange={e => setParam(p.key, e.target.value)}
                      className="w-full bg-dark-700 border border-dark-600 rounded-lg
                                 px-3 py-2 text-sm text-white"
                    >
                      {p.options.map(o => (
                        <option key={o} value={o}>{o}</option>
                      ))}
                    </select>
                  )}
                </div>
              ))}
            </div>
          )}

          {/* PCA toggle */}
          <div className="mt-5 pt-4 border-t border-dark-600 flex items-center gap-2">
            <input
              type="checkbox"
              id="pca"
              checked={usePCA}
              onChange={e => setUsePCA(e.target.checked)}
              className="accent-blue-500"
            />
            <label htmlFor="pca" className="text-sm text-gray-300 cursor-pointer">
              Apply PCA (5 components)
            </label>
          </div>
        </div>

        {/* Train button */}
        <button
          onClick={train}
          disabled={isTraining || !selected || loading}
          className="w-full py-3 rounded-xl font-bold text-sm transition-all
            bg-accent hover:bg-accent-hover text-white
            disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isTraining
            ? `Training… ${progress}%`
            : `Train ${activeModel?.name ?? ''}`}
        </button>

        {isTraining && (
          <div className="bg-dark-600 rounded-full h-2 overflow-hidden">
            <div
              className="bg-accent h-full rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        )}
      </div>

      {/* ── 3. Results ─────────────────────────────────────────────────────── */}
      <div className="space-y-4">
        {results ? (
          <div className="bg-dark-800 border border-dark-600 rounded-xl p-4 space-y-3">
            <div className="text-sm font-semibold text-gray-300">Results</div>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: 'R²',   value: results.r2,   color: 'text-blue-400'   },
                { label: 'RMSE', value: results.rmse, color: 'text-red-400'    },
                { label: 'MAE',  value: results.mae,  color: 'text-yellow-400' },
                { label: 'MSE',  value: results.mse,  color: 'text-purple-400' },
              ].map(m => (
                <div key={m.label} className="bg-dark-700 rounded-lg p-3 text-center">
                  <div className="text-xs text-gray-500">{m.label}</div>
                  <div className={`text-xl font-bold ${m.color}`}>
                    {typeof m.value === 'number' ? m.value.toFixed(4) : '—'}
                  </div>
                </div>
              ))}
            </div>
            <div className="text-xs text-gray-500 text-center">
              Logged to MLflow as {results.exp_id}
            </div>
          </div>
        ) : (
          <div className="bg-dark-800 border border-dark-600 rounded-xl p-4
            flex items-center justify-center h-48 text-gray-600 text-sm">
            Train a model to see results
          </div>
        )}
      </div>

    </div>
  )
}
