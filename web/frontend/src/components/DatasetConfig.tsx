'use client'

import { useState, useRef, useEffect } from 'react'
import { 
  Upload, X, FileText, Check, Trash2, RefreshCw, 
  Database, Cloud, FolderOpen, Loader2, AlertCircle,
  Plus, ExternalLink
} from 'lucide-react'

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

type DatasetSource = 'upload' | 'huggingface' | 'modelscope' | 'local_path'

interface Dataset {
  id: string
  name: string
  source: DatasetSource
  path: string
  subset?: string | null
  split?: string | null
  total_samples: number
  size_human: string
  format: string
  created_at: number
  selected: boolean
}

interface Props {
  selectedPaths: string[]
  onSelectionChange: (paths: string[]) => void
}

const SOURCE_TABS: { id: DatasetSource; label: string; icon: any; desc: string }[] = [
  { id: 'upload', label: 'Upload', icon: Upload, desc: 'Upload from your computer' },
  { id: 'huggingface', label: 'HuggingFace', icon: Cloud, desc: 'Use HuggingFace datasets' },
  { id: 'modelscope', label: 'ModelScope', icon: Cloud, desc: 'Use ModelScope datasets' },
  { id: 'local_path', label: 'Local Path', icon: FolderOpen, desc: 'Use server local path' },
]

const POPULAR_HF_DATASETS = [
  { id: 'tatsu-lab/alpaca', name: 'Alpaca', desc: '52K instruction-following' },
  { id: 'databricks/dolly-15k', name: 'Dolly 15K', desc: '15K instruction pairs' },
  { id: 'OpenAssistant/oasst1', name: 'OpenAssistant', desc: 'Human feedback data' },
  { id: 'timdettmers/openassistant-guanaco', name: 'Guanaco', desc: 'High-quality chat' },
]

export default function DatasetConfig({ selectedPaths, onSelectionChange }: Props) {
  const [activeTab, setActiveTab] = useState<DatasetSource>('upload')
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [isLoading, setIsLoading] = useState(false)
  
  // Upload state
  const [uploadName, setUploadName] = useState('')
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadError, setUploadError] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  
  // Register state (for HF/MS/Local)
  const [registerName, setRegisterName] = useState('')
  const [registerPath, setRegisterPath] = useState('')
  const [registerSubset, setRegisterSubset] = useState('')
  const [registerSplit, setRegisterSplit] = useState('train')
  const [isRegistering, setIsRegistering] = useState(false)
  const [registerError, setRegisterError] = useState('')
  
  // Delete confirmation
  const [deleteTarget, setDeleteTarget] = useState<Dataset | null>(null)
  const [deleteConfirm, setDeleteConfirm] = useState('')
  const [isDeleting, setIsDeleting] = useState(false)

  // Fetch datasets on mount
  useEffect(() => {
    fetchDatasets()
  }, [])

  const fetchDatasets = async () => {
    setIsLoading(true)
    try {
      const res = await fetch(`${API_URL}/api/datasets/list-all`)
      if (res.ok) {
        const data = await res.json()
        // Preserve selection state
        const newDatasets = data.datasets.map((ds: Dataset) => ({
          ...ds,
          selected: selectedPaths.includes(ds.path) || 
                    (datasets.find(d => d.id === ds.id)?.selected ?? true)
        }))
        setDatasets(newDatasets)
      }
    } catch (e) {
      console.error('Failed to fetch datasets:', e)
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setUploadFile(file)
      if (!uploadName) setUploadName(file.name.replace(/\.(jsonl|json|csv)$/i, ''))
    }
  }

  const checkNameAvailable = async (name: string): Promise<boolean> => {
    try {
      const res = await fetch(`${API_URL}/api/datasets/check-name?name=${encodeURIComponent(name)}`)
      if (res.ok) {
        const data = await res.json()
        return data.available
      }
    } catch (e) {
      console.error('Name check failed:', e)
    }
    return true // Allow if check fails
  }

  const uploadDataset = async () => {
    if (!uploadFile || !uploadName.trim()) {
      setUploadError('Please provide both a file and a name')
      return
    }
    setIsUploading(true)
    setUploadError('')
    try {
      // Check name availability first
      const nameAvailable = await checkNameAvailable(uploadName.trim())
      if (!nameAvailable) {
        setUploadError(`Dataset name "${uploadName}" is already in use. Please choose a different name.`)
        setIsUploading(false)
        return
      }

      const formData = new FormData()
      formData.append('file', uploadFile)
      const res = await fetch(`${API_URL}/api/datasets/upload?dataset_name=${encodeURIComponent(uploadName.trim())}`, {
        method: 'POST', body: formData
      })
      
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || `Upload failed with status ${res.status}`)
      }
      
      const data = await res.json()
      if (data.success) {
        setUploadName('')
        setUploadFile(null)
        if (fileInputRef.current) fileInputRef.current.value = ''
        await fetchDatasets()
      } else {
        setUploadError(data.detail || 'Upload failed')
      }
    } catch (e: any) {
      setUploadError(e.message || `Upload failed: ${e}`)
    } finally {
      setIsUploading(false)
    }
  }

  const registerDataset = async () => {
    if (!registerName.trim() || !registerPath.trim()) {
      setRegisterError('Please provide a name and dataset ID/path')
      return
    }
    setIsRegistering(true)
    setRegisterError('')
    try {
      // Check name availability first
      const nameAvailable = await checkNameAvailable(registerName.trim())
      if (!nameAvailable) {
        setRegisterError(`Dataset name "${registerName}" is already in use. Please choose a different name.`)
        setIsRegistering(false)
        return
      }

      const res = await fetch(`${API_URL}/api/datasets/register`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: registerName.trim(),
          source: activeTab,
          dataset_id: registerPath.trim(),
          subset: registerSubset.trim() || null,
          split: registerSplit || 'train'
        })
      })
      
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || `Registration failed with status ${res.status}`)
      }
      
      const data = await res.json()
      if (data.success) {
        setRegisterName('')
        setRegisterPath('')
        setRegisterSubset('')
        await fetchDatasets()
      } else {
        setRegisterError(data.detail || 'Registration failed')
      }
    } catch (e: any) {
      setRegisterError(e.message || `Registration failed: ${e}`)
    } finally {
      setIsRegistering(false)
    }
  }

  const toggleSelection = (dataset: Dataset) => {
    const updated = datasets.map(d => 
      d.id === dataset.id ? { ...d, selected: !d.selected } : d
    )
    setDatasets(updated)
    const paths = updated.filter(d => d.selected).map(d => d.path)
    onSelectionChange(paths)
  }

  const confirmDelete = async () => {
    if (!deleteTarget || deleteConfirm.toLowerCase() !== 'delete') return
    setIsDeleting(true)
    try {
      const res = await fetch(`${API_URL}/api/datasets/unregister/${encodeURIComponent(deleteTarget.id)}?confirm=delete`, {
        method: 'DELETE'
      })
      if (res.ok) {
        await fetchDatasets()
        setDeleteTarget(null)
        setDeleteConfirm('')
      }
    } catch (e) {
      console.error('Delete failed:', e)
    } finally {
      setIsDeleting(false)
    }
  }

  const selectQuickDataset = (datasetId: string, name: string) => {
    setRegisterPath(datasetId)
    setRegisterName(name)
  }

  const selectedCount = datasets.filter(d => d.selected).length

  const getSourceIcon = (source: DatasetSource) => {
    switch (source) {
      case 'upload': return <Upload className="w-4 h-4" />
      case 'huggingface': return <span className="text-xs">🤗</span>
      case 'modelscope': return <Cloud className="w-4 h-4" />
      case 'local_path': return <FolderOpen className="w-4 h-4" />
    }
  }

  const getSourceColor = (source: DatasetSource) => {
    switch (source) {
      case 'upload': return 'bg-blue-100 text-blue-700'
      case 'huggingface': return 'bg-yellow-100 text-yellow-700'
      case 'modelscope': return 'bg-purple-100 text-purple-700'
      case 'local_path': return 'bg-green-100 text-green-700'
    }
  }

  return (
    <div className="space-y-5">
      {/* Delete Confirmation Modal */}
      {deleteTarget && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl p-6 max-w-md w-full shadow-2xl">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 bg-red-100 rounded-full flex items-center justify-center">
                <Trash2 className="w-5 h-5 text-red-600" />
              </div>
              <div>
                <h3 className="font-bold text-slate-900">Remove Dataset</h3>
                <p className="text-sm text-slate-500">This will unregister the dataset</p>
              </div>
            </div>
            <div className="bg-slate-50 rounded-lg p-3 mb-4">
              <p className="text-sm text-slate-700"><strong>{deleteTarget.name}</strong></p>
              <p className="text-xs text-slate-500">{deleteTarget.source} • {deleteTarget.path}</p>
            </div>
            <p className="text-sm text-slate-600 mb-3">Type <strong className="text-red-600">delete</strong> to confirm:</p>
            <input type="text" value={deleteConfirm} onChange={(e) => setDeleteConfirm(e.target.value)}
              placeholder="Type 'delete'" autoFocus
              className="w-full px-3 py-2 border border-slate-300 rounded-lg mb-4" />
            <div className="flex gap-2">
              <button onClick={() => { setDeleteTarget(null); setDeleteConfirm('') }}
                className="flex-1 px-4 py-2 border border-slate-300 rounded-lg font-medium">Cancel</button>
              <button onClick={confirmDelete} disabled={deleteConfirm.toLowerCase() !== 'delete' || isDeleting}
                className="flex-1 px-4 py-2 bg-red-600 text-white rounded-lg font-medium disabled:opacity-50 flex items-center justify-center gap-2">
                {isDeleting ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}Delete
              </button>
            </div>
          </div>
        </div>
      )}

      <div>
        <h2 className="text-xl font-bold text-slate-900 mb-1">Configure Dataset</h2>
        <p className="text-slate-600 text-sm">Add datasets from multiple sources for training</p>
      </div>

      {/* Source Tabs */}
      <div className="flex flex-wrap gap-2 p-1 bg-slate-100 rounded-lg">
        {SOURCE_TABS.map(tab => (
          <button key={tab.id} onClick={() => setActiveTab(tab.id)}
            className={`flex-1 min-w-[120px] flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-all ${
              activeTab === tab.id ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
            }`}>
            <tab.icon className="w-4 h-4" />
            <span className="hidden sm:inline">{tab.label}</span>
          </button>
        ))}
      </div>

      {/* Add Dataset Form */}
      <div className="bg-slate-50 rounded-lg p-4 border border-slate-200">
        <h4 className="font-medium text-slate-900 mb-3 flex items-center gap-2">
          <Plus className="w-4 h-4" />
          Add {SOURCE_TABS.find(t => t.id === activeTab)?.label} Dataset
        </h4>

        {/* Upload Form */}
        {activeTab === 'upload' && (
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Dataset Name *</label>
              <input type="text" value={uploadName} onChange={(e) => setUploadName(e.target.value)}
                placeholder="My Training Dataset"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">File (.jsonl, .json, .csv) *</label>
              <input ref={fileInputRef} type="file" accept=".jsonl,.json,.csv" onChange={handleFileSelect}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg bg-white file:mr-3 file:px-3 file:py-1 file:border-0 file:bg-primary-100 file:text-primary-700 file:rounded file:font-medium file:text-sm" />
            </div>
            {uploadFile && <p className="text-sm text-slate-600">Selected: <strong>{uploadFile.name}</strong></p>}
            {uploadError && <p className="text-sm text-red-600 flex items-center gap-1"><AlertCircle className="w-4 h-4" />{uploadError}</p>}
            <button onClick={uploadDataset} disabled={!uploadFile || !uploadName.trim() || isUploading}
              className="w-full py-2 bg-primary-600 text-white rounded-lg font-medium disabled:opacity-50 flex items-center justify-center gap-2">
              {isUploading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Upload className="w-4 h-4" />}
              {isUploading ? 'Uploading...' : 'Upload Dataset'}
            </button>
          </div>
        )}

        {/* HuggingFace / ModelScope / Local Path Form */}
        {activeTab !== 'upload' && (
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">Dataset Name *</label>
              <input type="text" value={registerName} onChange={(e) => setRegisterName(e.target.value)}
                placeholder="My Dataset"
                className="w-full px-3 py-2 border border-slate-300 rounded-lg" />
            </div>
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                {activeTab === 'local_path' ? 'Local Path *' : 'Dataset ID *'}
              </label>
              <input type="text" value={registerPath} onChange={(e) => setRegisterPath(e.target.value)}
                placeholder={activeTab === 'local_path' ? '/path/to/dataset.jsonl' : 'organization/dataset-name'}
                className="w-full px-3 py-2 border border-slate-300 rounded-lg" />
            </div>
            {activeTab !== 'local_path' && (
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Subset (optional)</label>
                  <input type="text" value={registerSubset} onChange={(e) => setRegisterSubset(e.target.value)}
                    placeholder="default"
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Split</label>
                  <select value={registerSplit} onChange={(e) => setRegisterSplit(e.target.value)}
                    className="w-full px-3 py-2 border border-slate-300 rounded-lg">
                    <option value="train">train</option>
                    <option value="validation">validation</option>
                    <option value="test">test</option>
                  </select>
                </div>
              </div>
            )}
            {registerError && <p className="text-sm text-red-600 flex items-center gap-1"><AlertCircle className="w-4 h-4" />{registerError}</p>}
            <button onClick={registerDataset} disabled={!registerName.trim() || !registerPath.trim() || isRegistering}
              className="w-full py-2 bg-primary-600 text-white rounded-lg font-medium disabled:opacity-50 flex items-center justify-center gap-2">
              {isRegistering ? <Loader2 className="w-4 h-4 animate-spin" /> : <Plus className="w-4 h-4" />}
              {isRegistering ? 'Registering...' : 'Add Dataset'}
            </button>

            {/* Quick select for HuggingFace */}
            {activeTab === 'huggingface' && (
              <div className="pt-3 border-t border-slate-200">
                <p className="text-xs text-slate-500 mb-2">Popular datasets:</p>
                <div className="grid grid-cols-2 gap-2">
                  {POPULAR_HF_DATASETS.map(ds => (
                    <button key={ds.id} onClick={() => selectQuickDataset(ds.id, ds.name)}
                      className="p-2 text-left border border-slate-200 rounded-lg hover:border-primary-300 hover:bg-primary-50 transition-all">
                      <p className="text-sm font-medium text-slate-900">{ds.name}</p>
                      <p className="text-xs text-slate-500">{ds.desc}</p>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Dataset List */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h4 className="font-medium text-slate-900">
            Registered Datasets
            {selectedCount > 0 && <span className="text-primary-600 ml-2">({selectedCount} selected for training)</span>}
          </h4>
          <button onClick={fetchDatasets} disabled={isLoading}
            className="text-sm text-slate-500 hover:text-slate-700 flex items-center gap-1">
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} /> Refresh
          </button>
        </div>

        {isLoading ? (
          <div className="text-center py-8 text-slate-500">
            <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />Loading...
          </div>
        ) : datasets.length === 0 ? (
          <div className="text-center py-8 text-slate-500 bg-slate-50 rounded-lg border border-dashed border-slate-300">
            <Database className="w-10 h-10 mx-auto mb-2 opacity-50" />
            <p>No datasets registered yet</p>
            <p className="text-sm">Add a dataset using the form above</p>
          </div>
        ) : (
          <div className="space-y-2 max-h-72 overflow-y-auto">
            {datasets.map(dataset => (
              <div key={dataset.id} onClick={() => toggleSelection(dataset)}
                className={`flex items-center gap-3 p-3 rounded-lg border transition-all cursor-pointer ${
                  dataset.selected ? 'border-primary-500 bg-primary-50' : 'border-slate-200 hover:border-slate-300 bg-white'
                }`}>
                <div className={`w-5 h-5 rounded border-2 flex items-center justify-center flex-shrink-0 ${
                  dataset.selected ? 'bg-primary-600 border-primary-600' : 'border-slate-300'
                }`}>
                  {dataset.selected && <Check className="w-3 h-3 text-white" />}
                </div>
                
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${getSourceColor(dataset.source)}`}>
                  {getSourceIcon(dataset.source)}
                </div>
                
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-slate-900 truncate">{dataset.name}</p>
                  <p className="text-xs text-slate-500 truncate">
                    {dataset.path}
                    {dataset.subset && ` (${dataset.subset})`}
                    {dataset.split && ` [${dataset.split}]`}
                  </p>
                </div>
                
                <div className="text-right flex-shrink-0 hidden sm:block">
                  <p className="text-xs font-medium text-slate-600">{dataset.size_human}</p>
                  <p className="text-xs text-slate-400">{dataset.format}</p>
                </div>
                
                <button onClick={(e) => { e.stopPropagation(); setDeleteTarget(dataset) }}
                  className="p-2 text-slate-400 hover:text-red-600 hover:bg-red-50 rounded-lg flex-shrink-0">
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}

        {selectedCount === 0 && datasets.length > 0 && (
          <p className="text-sm text-amber-600 mt-2 flex items-center gap-1">
            <AlertCircle className="w-4 h-4" /> Please select at least one dataset for training
          </p>
        )}
      </div>
    </div>
  )
}
