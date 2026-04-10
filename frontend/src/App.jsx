import { Routes, Route, Navigate, NavLink } from 'react-router-dom'
import { LayoutDashboard, Brain, FlaskConical, Database, BarChart3, Zap } from 'lucide-react'
import Dashboard   from './pages/Dashboard'
import Training    from './pages/Training'
import Experiments from './pages/Experiments'
import DataPage    from './pages/DataPage'
import Comparison  from './pages/Comparison'
import Predict     from './pages/Predict'

const NAV = [
  { to:'/dashboard',   icon:LayoutDashboard, label:'Dashboard'   },
  { to:'/training',    icon:Brain,           label:'Training'    },
  { to:'/experiments', icon:FlaskConical,    label:'Experiments' },
  { to:'/data',        icon:Database,        label:'Data'        },
  { to:'/comparison',  icon:BarChart3,       label:'Comparison'  },
  { to:'/predict',     icon:Zap,             label:'Predict'     },
]

export default function App() {
  return (
    <div className="flex h-screen overflow-hidden">

      {/* Sidebar */}
      <aside className="w-52 bg-dark-800 border-r border-dark-600 flex flex-col shrink-0">
        <div className="px-4 py-5 border-b border-dark-600">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-accent flex items-center justify-center text-xs font-bold text-white">ML</div>
            <div>
              <div className="text-sm font-bold text-white">MLA Platform</div>
              <div className="text-xs text-gray-500">ER Wait Time</div>
            </div>
          </div>
        </div>
        <nav className="flex-1 p-3 space-y-1">
          {NAV.map(({ to, icon:Icon, label }) => (
            <NavLink key={to} to={to} className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-all
               ${isActive ? 'bg-accent/15 text-accent' : 'text-gray-400 hover:bg-dark-700 hover:text-white'}`
            }>
              <Icon size={15} />{label}
            </NavLink>
          ))}
        </nav>
        <div className="p-3 border-t border-dark-600 text-xs text-gray-600 text-center">
          scikit-learn + MLflow
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 overflow-y-auto p-6 bg-dark-900">
        <Routes>
          <Route path="/"            element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard"   element={<Dashboard />} />
          <Route path="/training"    element={<Training />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/data"        element={<DataPage />} />
          <Route path="/comparison"  element={<Comparison />} />
          <Route path="/predict"     element={<Predict />} />
        </Routes>
      </main>
    </div>
  )
}
