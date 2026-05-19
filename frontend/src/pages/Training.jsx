import { useState, useEffect } from 'react'
import { fetchModels, trainModel } from '../services/api'
import toast from 'react-hot-toast'

const HYPERPARAMS = {
  adaboost: [
    {
      key:'n_estimators',
      label:'Estimators',
      type:'range',
      min:10,
      max:500,
      step:10,
      default:100
    },
    {
      key:'learning_rate',
      label:'Learning Rate',
      type:'range',
      min:0.01,
      max:2,
      step:0.01,
      default:1.0
    },
    {
      key:'loss',
      label:'Loss',
      type:'select',
      options:['linear','square','exponential'],
      default:'linear'
    },
  ],

  xgboost: [
    {
      key:'n_estimators',
      label:'Trees',
      type:'range',
      min:10,
      max:1000,
      step:10,
      default:100
    },
    {
      key:'learning_rate',
      label:'Learning Rate',
      type:'range',
      min:0.01,
      max:0.5,
      step:0.01,
      default:0.1
    },
    {
      key:'max_depth',
      label:'Max Depth',
      type:'range',
      min:1,
      max:20,
      step:1,
      default:6
    },
    {
      key:'subsample',
      label:'Subsample',
      type:'range',
      min:0.5,
      max:1,
      step:0.05,
      default:1
    },
    {
      key:'colsample_bytree',
      label:'Feature Sample',
      type:'range',
      min:0.5,
      max:1,
      step:0.05,
      default:1
    },
    {
      key:'gamma',
      label:'Gamma',
      type:'range',
      min:0,
      max:10,
      step:0.1,
      default:0
    },
  ],

  random_forest: [
    {
      key:'n_estimators',
      label:'Trees',
      type:'range',
      min:10,
      max:1000,
      step:10,
      default:100
    },
    {
      key:'max_depth',
      label:'Max Depth',
      type:'range',
      min:1,
      max:100,
      step:1,
      default:10
    },
    {
      key:'min_samples_split',
      label:'Min Split',
      type:'range',
      min:2,
      max:20,
      step:1,
      default:2
    },
    {
      key:'min_samples_leaf',
      label:'Min Leaf',
      type:'range',
      min:1,
      max:20,
      step:1,
      default:1
    },
    {
      key:'max_features',
      label:'Max Features',
      type:'select',
      options:['sqrt','log2'],
      default:'sqrt'
    },
    {
      key:'bootstrap',
      label:'Bootstrap',
      type:'select',
      options:['true','false'],
      default:true
    },
  ],

  svr: [
    {
      key:'C',
      label:'C',
      type:'range',
      min:0.1,
      max:100,
      step:0.1,
      default:1
    },
    {
      key:'kernel',
      label:'Kernel',
      type:'select',
      options:['rbf','linear','poly','sigmoid'],
      default:'rbf'
    },
    {
      key:'gamma',
      label:'Gamma',
      type:'select',
      options:['scale','auto'],
      default:'scale'
    },
    {
      key:'epsilon',
      label:'Epsilon',
      type:'range',
      min:0.01,
      max:1,
      step:0.01,
      default:0.1
    },
  ],

  knn: [
    {
      key:'n_neighbors',
      label:'Neighbors',
      type:'range',
      min:1,
      max:50,
      step:1,
      default:5
    },
    {
      key:'weights',
      label:'Weights',
      type:'select',
      options:['uniform','distance'],
      default:'uniform'
    },
    {
      key:'metric',
      label:'Metric',
      type:'select',
      options:['euclidean','manhattan','minkowski'],
      default:'minkowski'
    },
  ],

  linear_reg: [
    {
      key:'alpha',
      label:'Alpha',
      type:'range',
      min:0.001,
      max:100,
      step:0.001,
      default:1
    },
    {
      key:'solver',
      label:'Solver',
      type:'select',
      options:[
        'auto',
        'svd',
        'cholesky',
        'lsqr',
        'sag',
        'saga'
      ],
      default:'auto'
    },
  ],

  neural_net: [
    {
      key:'learning_rate_init',
      label:'Learning Rate',
      type:'range',
      min:0.0001,
      max:0.1,
      step:0.0001,
      default:0.001
    },
    {
      key:'hidden_layer_sizes',
      label:'Architecture',
      type:'select',
      options:[
        '(64,)',
        '(128,)',
        '(128,64)',
        '(256,128)',
        '(256,128,64)'
      ],
      default:'(128,64)'
    },
    {
      key:'activation',
      label:'Activation',
      type:'select',
      options:['relu','tanh','logistic'],
      default:'relu'
    },
    {
      key:'solver',
      label:'Optimizer',
      type:'select',
      options:['adam','sgd'],
      default:'adam'
    },
    {
      key:'batch_size',
      label:'Batch Size',
      type:'range',
      min:16,
      max:256,
      step:16,
      default:32
    },
    {
      key:'max_iter',
      label:'Epochs',
      type:'range',
      min:100,
      max:1000,
      step:50,
      default:300
    },
    {
      key:'alpha',
      label:'L2 Regularization',
      type:'range',
      min:0.00001,
      max:0.01,
      step:0.00001,
      default:0.0001
    },
  ],
}

export default function Training() {
  const [models,      setModels]      = useState([])
  const [selected,    setSelected]    = useState('random_forest')
  const [params,      setParams]      = useState({})
  const [usePCA,      setUsePCA]      = useState(false)
  const [isTraining,  setIsTraining]  = useState(false)
  const [progress,    setProgress]    = useState(0)
  const [results,     setResults]     = useState(null)

  useEffect(() => {
    fetchModels().then(setModels).catch(() => {})
  }, [])

  const setParam = (k, v) => setParams(p => ({ ...p, [k]: v }))

  const train = async () => {
    setIsTraining(true)
    setProgress(0)
    setResults(null)

    // Simulate progress
    const iv = setInterval(() => {
      setProgress(p => { if (p >= 90) { clearInterval(iv); return 90 } return p + 10 })
    }, 400)

    const id = toast.loading(`Training ${selected}...`)
    try {
      const finalParams = { ...params }
        // Convert tuple string for MLP
        if (selected === 'neural_net' && finalParams.hidden_layer_sizes) {
        finalParams.hidden_layer_sizes =
            JSON.parse(
            finalParams.hidden_layer_sizes
                .replace(/\(/g, '[')
                .replace(/\)/g, ']')
            )
        }

        // Convert boolean select
        finalParams.bootstrap = Boolean(finalParams.bootstrap)

        const res = await trainModel({
        model: selected,
        hyperparams: finalParams,
        use_pca: usePCA
        })
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

  const hparams = HYPERPARAMS[selected] || []

  return (
    <div className="grid grid-cols-3 gap-5">

      {/* Model selector */}
      <div className="space-y-4">
        <div className="bg-dark-800 border border-dark-600 rounded-xl p-4">
          <div className="text-sm font-semibold text-gray-300 mb-3">Select Model</div>
          <div className="space-y-2">
            {(models.length ? models : [
              {key:'random_forest',label:'Random Forest',family:'Ensemble'},
              {key:'svr',          label:'SVR',          family:'Kernel'},
              {key:'knn',          label:'KNN',          family:'Instance'},
              {key:'linear_reg',   label:'Ridge Regression', family:'Linear'},
              {key:'neural_net',   label:'Neural Network',family:'Deep Learning'},
            ]).map(m => (
              <div key={m.key}
                onClick={() => setSelected(m.key)}
                className={`cursor-pointer rounded-lg border p-3 transition-all
                  ${selected===m.key
                    ? 'border-accent/60 bg-accent/10'
                    : 'border-dark-600 hover:border-dark-500'}`}
              >
                <div className="text-sm font-semibold text-white">{m.name || m.label}</div>
                <div className="text-xs text-gray-500">{m.family}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Hyperparams */}
      <div className="space-y-4">
        <div className="bg-dark-800 border border-dark-600 rounded-xl p-4">
          <div className="text-sm font-semibold text-gray-300 mb-4">Hyperparameters</div>
          <div className="space-y-4">
            {hparams.map(p => (
              <div key={p.key}>
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-gray-300">{p.label}</span>
                  <span className="text-accent font-mono">{params[p.key] ?? p.default}</span>
                </div>
                {p.type === 'range' ? (
                  <input type="range" min={p.min} max={p.max} step={p.step}
                    defaultValue={p.default}
                    onChange={e => setParam(p.key, parseFloat(e.target.value))}
                    className="w-full accent-blue-500 cursor-pointer" />
                ) : (
                  <select onChange={e => setParam(p.key, e.target.value)} defaultValue={p.default}
                    className="w-full bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm text-white">
                    {p.options.map(o => <option key={o}>{o}</option>)}
                  </select>
                )}
              </div>
            ))}
          </div>

          {/* PCA toggle */}
          <div className="mt-4 flex items-center gap-2">
            <input type="checkbox" id="pca" checked={usePCA}
              onChange={e => setUsePCA(e.target.checked)}
              className="accent-blue-500" />
            <label htmlFor="pca" className="text-sm text-gray-300 cursor-pointer">
              Apply PCA (5 components)
            </label>
          </div>
        </div>

        {/* Train button */}
        <button onClick={train} disabled={isTraining}
          className="w-full py-3 rounded-xl font-bold text-sm transition-all
            bg-accent hover:bg-accent-hover text-white disabled:opacity-50 disabled:cursor-not-allowed">
          {isTraining ? `Training... ${progress}%` : `Train ${selected}`}
        </button>

        {isTraining && (
          <div className="bg-dark-600 rounded-full h-2 overflow-hidden">
            <div className="bg-accent h-full rounded-full transition-all duration-300"
              style={{width:`${progress}%`}} />
          </div>
        )}
      </div>

      {/* Results */}
      <div className="space-y-4">
        {results ? (
          <div className="bg-dark-800 border border-dark-600 rounded-xl p-4 space-y-3">
            <div className="text-sm font-semibold text-gray-300">Results</div>
            <div className="grid grid-cols-2 gap-3">
              {[
                { label:'R²',   value:results.r2,   color:'text-blue-400'    },
                { label:'RMSE', value:results.rmse, color:'text-red-400'     },
                { label:'MAE',  value:results.mae,  color:'text-yellow-400'  },
                { label:'MSE',  value:results.mse,  color:'text-purple-400'  },
              ].map(m => (
                <div key={m.label} className="bg-dark-700 rounded-lg p-3 text-center">
                  <div className="text-xs text-gray-500">{m.label}</div>
                  <div className={`text-xl font-bold ${m.color}`}>{m.value}</div>
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
