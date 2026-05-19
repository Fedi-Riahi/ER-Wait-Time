import { useState, useEffect, useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { fetchPreview, uploadDataset } from '../services/api'
import { Upload } from 'lucide-react'
import toast from 'react-hot-toast'

export default function DataPage() {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchPreview().then(setStats).catch(() => setLoading(false)).finally(() => setLoading(false))
  }, [])

  const onDrop = useCallback(([file]) => {
    if (!file) return
    const fd = new FormData()
    fd.append('file', file)
    toast.loading('Uploading...')
    uploadDataset(fd)
      .then(res => { setStats(res); toast.dismiss(); toast.success(`Loaded ${file.name}`) })
      .catch(() => toast.error('Upload failed'))
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop, accept: { 'text/csv': ['.csv'] }
  })

  const cols = stats?.preview?.[0] ? Object.keys(stats.preview[0]) : []

  return (
    <div className="space-y-5">
      <h1 className="text-xl font-bold text-white">Dataset</h1>

      {/* Drop zone */}
      <div {...getRootProps()}
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-all
          ${isDragActive ? 'border-accent bg-accent/10' : 'border-dark-600 hover:border-accent/40'}`}>
        <input {...getInputProps()} />
        <Upload size={28} className="mx-auto text-gray-600 mb-3" />
        <div className="text-sm text-gray-300">
          {isDragActive ? 'Drop CSV here' : 'Drag & drop your CSV, or click to browse'}
        </div>
      </div>

      {/* Stats */}
      {stats && (
        <>
          <div className="grid grid-cols-4 gap-4">
            {[
              { label:'Rows', value:stats.rows?.toLocaleString(), color:'text-blue-400' },
              { label:'Columns', value:stats.columns, color:'text-purple-400' },
              { label:'Nulls', value:stats.null_count?.toLocaleString(), color:stats.null_count > 0 ? 'text-yellow-400' : 'text-emerald-400' },
              { label:'Target', value:'wait_time_min', color:'text-accent' },
            ].map(s => (
              <div key={s.label} className="bg-dark-800 border border-dark-600 rounded-xl p-4 text-center">
                <div className="text-xs text-gray-500 mb-1">{s.label}</div>
                <div className={`text-lg font-bold ${s.color}`}>{s.value}</div>
              </div>
            ))}
          </div>

          {/* Preview table */}
          <div className="bg-dark-800 border border-dark-600 rounded-xl p-5">
            <div className="text-sm font-semibold text-gray-300 mb-4">Preview (first 10 rows)</div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs min-w-max">
                <thead>
                  <tr className="border-b border-dark-600 text-gray-500">
                    {cols.map(c => (
                      <th key={c} className="pb-2 pr-4 text-left font-medium whitespace-nowrap">
                        {c === 'wait_time_min'
                          ? <span className="text-accent font-bold">⭐ {c}</span>
                          : c}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-dark-600">
                  {stats.preview.map((row, i) => (
                    <tr key={i} className="text-gray-300 hover:bg-dark-700">
                      {cols.map(c => (
                        <td key={c} className="py-1.5 pr-4 font-mono whitespace-nowrap">
                          {row[c] === 'NULL'
                            ? <span className="text-red-500/70">NULL</span>
                            : String(row[c])}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
