import { useState } from 'react'
import { predict } from '../services/api'
import { Zap } from 'lucide-react'
import toast from 'react-hot-toast'

const FIELDS = [
  { key:'triage_level_ESI',   label:'Triage Level ESI (1–5)',  placeholder:'3'   },
  { key:'doctors_available',  label:'Doctors Available',        placeholder:'4'   },
  { key:'patients_waiting',   label:'Patients Waiting',         placeholder:'18'  },
  { key:'age',                label:'Patient Age',              placeholder:'34'  },
  { key:'heart_rate_bpm',     label:'Heart Rate (bpm)',         placeholder:'88'  },
  { key:'systolic_bp_mmhg',   label:'Systolic BP (mmHg)',       placeholder:'120' },
  { key:'spo2_pct',           label:'SpO2 (%)',                 placeholder:'97'  },
  { key:'occupancy_rate_pct', label:'Occupancy Rate (%)',       placeholder:'72'  },
  { key:'is_rush_hour',       label:'Rush Hour (0 or 1)',       placeholder:'1'   },
  { key:'is_weekend',         label:'Weekend (0 or 1)',         placeholder:'0'   },
]

export default function Predict() {
  const [form,    setForm]    = useState({})
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(false)
  const [model,   setModel]   = useState('random_forest')

  const run = async () => {
    setLoading(true)
    try {
      const res = await predict({ model, inputs: form })
      setResult(res)
    } catch {
      toast.error('Prediction failed — is the backend running?')
    } finally {
      setLoading(false)
    }
  }

  const category = result == null ? null
    : result.wait_time_min < 15 ? { label:'Immediate', color:'text-red-400',    bg:'bg-red-500/10'    }
    : result.wait_time_min < 45 ? { label:'Urgent',    color:'text-yellow-400', bg:'bg-yellow-500/10' }
    :                              { label:'Standard',  color:'text-blue-400',   bg:'bg-blue-500/10'   }

  return (
    <div className="grid grid-cols-2 gap-6">

      {/* Form */}
      <div className="bg-dark-800 border border-dark-600 rounded-xl p-5">
        <h2 className="text-sm font-semibold text-gray-300 mb-1">Predict Wait Time</h2>
        <p className="text-xs text-gray-500 mb-4">Fill in patient & context data</p>

        {/* Model selector */}
        <div className="mb-4">
          <label className="block text-xs text-gray-400 mb-1">Model</label>
          <select
            value={model}
            onChange={e => setModel(e.target.value)}
            className="w-full bg-dark-700 border border-dark-600 rounded-lg px-3 py-2 text-sm text-white"
            >
            <option value="linear_reg">Linear Regression ⭐ Best</option>
            <option value="xgboost">XGBoost 🥈</option>
            <option value="random_forest">Random Forest</option>
            <option value="svr">SVR</option>
            <option value="knn">KNN</option>
            <option value="neural_net">Neural Network</option>
            <option value="adaboost">AdaBoost</option>
            </select>
        </div>

        <div className="space-y-3">
          {FIELDS.map(f => (
            <div key={f.key}>
              <label className="block text-xs text-gray-400 mb-1">{f.label}</label>
              <input
                type="number"
                placeholder={f.placeholder}
                onChange={e => setForm(p => ({ ...p, [f.key]: parseFloat(e.target.value) }))}
                className="w-full bg-dark-700 border border-dark-600 rounded-lg px-3 py-2
                           text-sm text-white placeholder-gray-700 focus:border-accent focus:outline-none"
              />
            </div>
          ))}
        </div>

        <button onClick={run} disabled={loading}
          className="w-full mt-4 py-3 rounded-xl bg-accent hover:bg-accent-hover
                     text-white font-bold text-sm transition-all disabled:opacity-50 flex items-center justify-center gap-2">
          <Zap size={16} />
          {loading ? 'Predicting...' : 'Predict Wait Time'}
        </button>
      </div>

      {/* Result */}
      <div className="space-y-4">
        {result ? (
          <>
            <div className={`${category.bg} border border-dark-600 rounded-xl p-10 text-center`}>
              <div className="text-sm text-gray-400 mb-2">Predicted Wait Time</div>
              <div className={`text-7xl font-extrabold ${category.color}`}>
                {result.wait_time_min}
              </div>
              <div className="text-gray-400 mt-2">minutes</div>
              <div className={`inline-block mt-3 px-3 py-1 rounded-full text-xs font-semibold ${category.color} bg-dark-700`}>
                {category.label}
              </div>
              {result.warning && (
                <div className="mt-3 text-xs text-yellow-500">{result.warning}</div>
              )}
            </div>

            <div className="bg-dark-800 border border-dark-600 rounded-xl p-4">
              <div className="text-xs font-semibold text-gray-400 mb-3">Input Summary</div>
              <div className="grid grid-cols-2 gap-1 text-xs">
                {Object.entries(form).map(([k,v]) => (
                  <div key={k} className="flex justify-between border-b border-dark-600 py-1">
                    <span className="text-gray-500">{k}</span>
                    <span className="text-gray-300 font-mono">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </>
        ) : (
          <div className="bg-dark-800 border border-dark-600 rounded-xl
            flex items-center justify-center h-64 text-gray-600 text-sm text-center px-6">
            <div>
              <Zap size={36} className="mx-auto mb-3 text-gray-700" />
              Fill in the form and click Predict
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
