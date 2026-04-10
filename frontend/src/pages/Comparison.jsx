import { useState, useEffect } from 'react'
import { fetchComparison } from '../services/api'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts'

const TT = { contentStyle:{ background:'#12121a', border:'1px solid #1e1e2e', borderRadius:8, color:'#e2e8f0' } }

export default function Comparison() {
  const [data, setData] = useState([])

  useEffect(() => { fetchComparison().then(setData).catch(() => {}) }, [])

  const chartData = data.slice(0,10).map(r => ({
    name: r.Experiment?.split('_').slice(0,2).join(' ') || r.Model,
    r2:   Math.round((r['R²'] || 0) * 100),
    rmse: r.RMSE || 0,
    mae:  r.MAE  || 0,
  }))

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-white">Model Comparison</h1>
        <p className="text-sm text-gray-500">All experiments ranked by R²</p>
      </div>

      {/* Bar charts */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-dark-800 border border-dark-600 rounded-xl p-5">
          <div className="text-sm font-semibold text-gray-300 mb-4">R² Score (%)</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
              <XAxis type="number" domain={[0,100]} stroke="#475569" tick={{fill:'#64748b',fontSize:11}} />
              <YAxis type="category" dataKey="name" stroke="#475569" tick={{fill:'#64748b',fontSize:10}} width={120} />
              <Tooltip {...TT} />
              <Bar dataKey="r2" fill="#3b82f6" radius={[0,4,4,0]} name="R² %" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-dark-800 border border-dark-600 rounded-xl p-5">
          <div className="text-sm font-semibold text-gray-300 mb-4">RMSE (lower = better)</div>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={chartData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" />
              <XAxis type="number" stroke="#475569" tick={{fill:'#64748b',fontSize:11}} />
              <YAxis type="category" dataKey="name" stroke="#475569" tick={{fill:'#64748b',fontSize:10}} width={120} />
              <Tooltip {...TT} />
              <Bar dataKey="rmse" fill="#ef4444" radius={[0,4,4,0]} name="RMSE" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Table */}
      <div className="bg-dark-800 border border-dark-600 rounded-xl p-5">
        <div className="text-sm font-semibold text-gray-300 mb-4">Full Comparison Table</div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm min-w-max">
            <thead>
              <tr className="text-xs text-gray-500 border-b border-dark-600">
                {['#','Experiment','Model','MAE','MSE','RMSE','R²'].map(h => (
                  <th key={h} className="pb-2 text-left pr-4 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-dark-600">
              {data.map((r, i) => (
                <tr key={i} className={`text-gray-300 text-xs ${i===0?'bg-emerald-500/5':''}`}>
                  <td className="py-2 pr-4 text-gray-600">{i+1}</td>
                  <td className="py-2 pr-4 font-mono text-gray-400">{r.Experiment}</td>
                  <td className="py-2 pr-4 font-semibold text-white">{r.Model}</td>
                  <td className="py-2 pr-4 text-yellow-400">{r.MAE}</td>
                  <td className="py-2 pr-4 text-purple-400">{r.MSE}</td>
                  <td className="py-2 pr-4 text-red-400">{r.RMSE}</td>
                  <td className="py-2 pr-4">
                    <span className={`font-bold ${i===0?'text-emerald-400':'text-blue-400'}`}>
                      {r['R²']} {i===0 && '⭐'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
