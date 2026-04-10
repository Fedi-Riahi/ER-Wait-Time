import { useState, useEffect } from 'react'
import { fetchSummary, fetchRuns } from '../services/api'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

const TT = { contentStyle:{ background:'#12121a', border:'1px solid #1e1e2e', borderRadius:8, color:'#e2e8f0' } }

export default function Dashboard() {
  const [summary, setSummary] = useState(null)
  const [runs,    setRuns]    = useState([])

  useEffect(() => {
    fetchSummary().then(setSummary).catch(() => {})
    fetchRuns().then(setRuns).catch(() => {})
  }, [])

  const chartData = runs.slice(0,8).map(r => ({
    name: r.name?.replace('_',' ').slice(0,15) || r.id,
    r2:   r.metrics.r2 * 100,
    rmse: r.metrics.rmse,
  }))

  const kpis = [
    { label:'Total Runs',   value: summary?.total_runs ?? '—', color:'text-blue-400',    icon:'🧪' },
    { label:'Best R²',      value: summary?.best_r2    ?? '—', color:'text-emerald-400', icon:'🎯' },
    { label:'Best RMSE',    value: summary?.best_rmse  ?? '—', color:'text-red-400',     icon:'📉' },
    { label:'Best Model',   value: summary?.best_model ?? '—', color:'text-yellow-400',  icon:'🏆' },
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-white">Dashboard</h1>
        <p className="text-sm text-gray-500">ER Wait Time — Experiment Overview</p>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-4 gap-4">
        {kpis.map(k => (
          <div key={k.label} className="bg-dark-800 border border-dark-600 rounded-xl p-4">
            <div className="flex justify-between items-start">
              <div>
                <div className="text-xs text-gray-500 mb-1">{k.label}</div>
                <div className={`text-2xl font-bold ${k.color}`}>{k.value}</div>
              </div>
              <span className="text-2xl">{k.icon}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-dark-800 border border-dark-600 rounded-xl p-5">
          <div className="text-sm font-semibold text-gray-300 mb-4">R² by Experiment</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
              <XAxis dataKey="name" stroke="#475569" tick={{fill:'#64748b',fontSize:10}} />
              <YAxis stroke="#475569" tick={{fill:'#64748b',fontSize:11}} domain={[0,100]} />
              <Tooltip {...TT} />
              <Bar dataKey="r2" fill="#3b82f6" radius={[4,4,0,0]} name="R² %" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-dark-800 border border-dark-600 rounded-xl p-5">
          <div className="text-sm font-semibold text-gray-300 mb-4">RMSE by Experiment</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
              <XAxis dataKey="name" stroke="#475569" tick={{fill:'#64748b',fontSize:10}} />
              <YAxis stroke="#475569" tick={{fill:'#64748b',fontSize:11}} />
              <Tooltip {...TT} />
              <Bar dataKey="rmse" fill="#ef4444" radius={[4,4,0,0]} name="RMSE (min)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recent runs table */}
      <div className="bg-dark-800 border border-dark-600 rounded-xl p-5">
        <div className="text-sm font-semibold text-gray-300 mb-4">Recent Runs</div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-gray-500 border-b border-dark-600">
              {['Run Name','Model','MAE','RMSE','R²','Status'].map(h => (
                <th key={h} className="pb-2 text-left font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-dark-600">
            {runs.slice(0,7).map(r => (
              <tr key={r.id} className="text-gray-300 text-xs">
                <td className="py-2 font-mono text-gray-400">{r.name || r.id}</td>
                <td className="py-2 font-medium text-white">{r.model}</td>
                <td className="py-2 text-yellow-400">{r.metrics.mae}</td>
                <td className="py-2 text-red-400">{r.metrics.rmse}</td>
                <td className="py-2 text-blue-400 font-bold">{r.metrics.r2}</td>
                <td className="py-2">
                  <span className={`px-2 py-0.5 rounded text-xs font-semibold
                    ${r.status === 'best'
                      ? 'bg-emerald-500/20 text-emerald-400'
                      : 'bg-gray-500/20 text-gray-400'}`}>
                    {r.status === 'best' ? '⭐ Best' : 'Done'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
