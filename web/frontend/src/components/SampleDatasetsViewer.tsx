'use client'

import { useState, useEffect } from 'react'
import { 
  Download, X, BookOpen, ChevronDown, ChevronUp, 
  FileText, CheckCircle2, Code, Loader2, ExternalLink,
  Copy, Check, Eye, FileCode, AlertCircle
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface SampleDataset {
  type: string
  filename: string
  display_name: string
  description: string
  format_description: string
  format_example: string
  compatible_methods: string[]
  compatible_algorithms: string[]
  usf_bios_verified: boolean
  preview: any[]
  download_available: boolean
  documentation_available: boolean
}

interface Props {
  isOpen: boolean
  onClose: () => void
}

export default function SampleDatasetsViewer({ isOpen, onClose }: Props) {
  const [samples, setSamples] = useState<SampleDataset[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [selectedSample, setSelectedSample] = useState<SampleDataset | null>(null)
  const [documentation, setDocumentation] = useState<string>('')
  const [isLoadingDocs, setIsLoadingDocs] = useState(false)
  const [activeTab, setActiveTab] = useState<'overview' | 'preview' | 'docs'>('overview')
  const [copiedExample, setCopiedExample] = useState(false)
  const [filterMethod, setFilterMethod] = useState<string>('all')

  useEffect(() => {
    if (isOpen && samples.length === 0) {
      fetchSamples()
    }
  }, [isOpen])

  const fetchSamples = async () => {
    setIsLoading(true)
    try {
      const res = await fetch('/api/datasets/sample-datasets')
      if (res.ok) {
        const data = await res.json()
        setSamples(data.samples || [])
      }
    } catch (e) {
      console.error('Failed to fetch samples:', e)
    } finally {
      setIsLoading(false)
    }
  }

  const fetchDocumentation = async (datasetType: string) => {
    setIsLoadingDocs(true)
    setDocumentation('')
    try {
      const res = await fetch(`/api/datasets/sample-datasets/${datasetType}/docs`)
      if (res.ok) {
        const data = await res.json()
        setDocumentation(data.documentation || '')
      }
    } catch (e) {
      console.error('Failed to fetch docs:', e)
    } finally {
      setIsLoadingDocs(false)
    }
  }

  const downloadSample = async (sample: SampleDataset) => {
    try {
      const res = await fetch(`/api/datasets/sample-datasets/${sample.type}/download`)
      if (res.ok) {
        const blob = await res.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = sample.filename
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      }
    } catch (e) {
      console.error('Failed to download:', e)
    }
  }

  const copyExample = (text: string) => {
    navigator.clipboard.writeText(text)
    setCopiedExample(true)
    setTimeout(() => setCopiedExample(false), 2000)
  }

  const selectSample = (sample: SampleDataset) => {
    setSelectedSample(sample)
    setActiveTab('overview')
    if (sample.documentation_available) {
      fetchDocumentation(sample.type)
    }
  }

  const filteredSamples = filterMethod === 'all' 
    ? samples 
    : samples.filter(s => s.compatible_methods.includes(filterMethod))

  const getMethodColor = (method: string) => {
    switch (method) {
      case 'sft': return 'bg-blue-100 text-blue-700 border-blue-200'
      case 'pt': return 'bg-green-100 text-green-700 border-green-200'
      case 'rlhf': return 'bg-purple-100 text-purple-700 border-purple-200'
      default: return 'bg-slate-100 text-slate-700 border-slate-200'
    }
  }

  const getAlgoColor = (algo: string) => {
    const colors: Record<string, string> = {
      dpo: 'bg-violet-100 text-violet-700',
      orpo: 'bg-fuchsia-100 text-fuchsia-700',
      simpo: 'bg-pink-100 text-pink-700',
      kto: 'bg-amber-100 text-amber-700',
      cpo: 'bg-rose-100 text-rose-700',
      rm: 'bg-cyan-100 text-cyan-700',
      ppo: 'bg-orange-100 text-orange-700',
      grpo: 'bg-lime-100 text-lime-700',
      gkd: 'bg-teal-100 text-teal-700',
    }
    return colors[algo] || 'bg-slate-100 text-slate-700'
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-slate-200 bg-gradient-to-r from-blue-50 to-indigo-50">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center">
              <BookOpen className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-slate-900">Dataset Format Examples</h2>
              <p className="text-sm text-slate-600">Download examples and read documentation for each format</p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-slate-200 rounded-lg transition-colors">
            <X className="w-5 h-5 text-slate-600" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 flex overflow-hidden">
          {/* Left Panel - Sample List */}
          <div className="w-80 border-r border-slate-200 flex flex-col bg-slate-50">
            {/* Filter */}
            <div className="p-3 border-b border-slate-200">
              <label className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2 block">
                Filter by Training Method
              </label>
              <div className="flex flex-wrap gap-1">
                {['all', 'sft', 'pt', 'rlhf'].map(method => (
                  <button
                    key={method}
                    onClick={() => setFilterMethod(method)}
                    className={`px-2.5 py-1 text-xs font-medium rounded-lg transition-colors ${
                      filterMethod === method
                        ? 'bg-blue-600 text-white'
                        : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
                    }`}
                  >
                    {method === 'all' ? 'All' : method.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>

            {/* Sample List */}
            <div className="flex-1 overflow-y-auto p-2 space-y-1">
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <Loader2 className="w-6 h-6 animate-spin text-blue-500" />
                </div>
              ) : filteredSamples.length === 0 ? (
                <div className="text-center py-8 text-slate-500 text-sm">No datasets found</div>
              ) : (
                filteredSamples.map(sample => (
                  <button
                    key={sample.type}
                    onClick={() => selectSample(sample)}
                    className={`w-full text-left p-3 rounded-lg transition-all ${
                      selectedSample?.type === sample.type
                        ? 'bg-blue-100 border-blue-300 border-2'
                        : 'bg-white border border-slate-200 hover:border-blue-200 hover:bg-blue-50'
                    }`}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="font-medium text-sm text-slate-900 truncate">
                          {sample.display_name}
                        </div>
                        <div className="flex items-center gap-1 mt-1 flex-wrap">
                          {sample.compatible_methods.map(m => (
                            <span key={m} className={`text-[10px] px-1.5 py-0.5 rounded font-medium uppercase ${getMethodColor(m)}`}>
                              {m}
                            </span>
                          ))}
                        </div>
                      </div>
                      {sample.usf_bios_verified && (
                        <CheckCircle2 className="w-4 h-4 text-green-500 flex-shrink-0" />
                      )}
                    </div>
                  </button>
                ))
              )}
            </div>
          </div>

          {/* Right Panel - Details */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {selectedSample ? (
              <>
                {/* Sample Header */}
                <div className="p-4 border-b border-slate-200 bg-white">
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="text-lg font-bold text-slate-900">{selectedSample.display_name}</h3>
                      <p className="text-sm text-slate-600 mt-1">{selectedSample.description}</p>
                      <div className="flex items-center gap-2 mt-2 flex-wrap">
                        {selectedSample.compatible_algorithms.map(algo => (
                          <span key={algo} className={`text-xs px-2 py-0.5 rounded-full font-medium ${getAlgoColor(algo)}`}>
                            {algo.toUpperCase()}
                          </span>
                        ))}
                        {selectedSample.usf_bios_verified && (
                          <span className="text-xs px-2 py-0.5 rounded-full bg-green-100 text-green-700 flex items-center gap-1">
                            <CheckCircle2 className="w-3 h-3" />
                            USF-BIOS Verified
                          </span>
                        )}
                      </div>
                    </div>
                    <button
                      onClick={() => downloadSample(selectedSample)}
                      disabled={!selectedSample.download_available}
                      className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg font-medium hover:bg-blue-700 transition-colors disabled:opacity-50"
                    >
                      <Download className="w-4 h-4" />
                      Download
                    </button>
                  </div>
                </div>

                {/* Tabs */}
                <div className="flex border-b border-slate-200 bg-white px-4">
                  {[
                    { id: 'overview', label: 'Overview', icon: Eye },
                    { id: 'preview', label: 'Preview', icon: FileCode },
                    { id: 'docs', label: 'Documentation', icon: BookOpen },
                  ].map(tab => (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id as any)}
                      className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
                        activeTab === tab.id
                          ? 'border-blue-600 text-blue-600'
                          : 'border-transparent text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      <tab.icon className="w-4 h-4" />
                      {tab.label}
                    </button>
                  ))}
                </div>

                {/* Tab Content */}
                <div className="flex-1 overflow-y-auto p-4 bg-slate-50">
                  {activeTab === 'overview' && (
                    <div className="space-y-4">
                      {/* Format Structure */}
                      <div className="bg-white rounded-lg border border-slate-200 p-4">
                        <h4 className="font-semibold text-slate-900 mb-2 flex items-center gap-2">
                          <Code className="w-4 h-4 text-blue-600" />
                          Format Structure
                        </h4>
                        <p className="text-sm text-slate-600 mb-3">{selectedSample.format_description}</p>
                        {selectedSample.format_example && (
                          <div className="relative">
                            <pre className="bg-slate-900 text-slate-100 p-3 rounded-lg text-xs overflow-x-auto">
                              <code>{selectedSample.format_example}</code>
                            </pre>
                            <button
                              onClick={() => copyExample(selectedSample.format_example)}
                              className="absolute top-2 right-2 p-1.5 bg-slate-700 hover:bg-slate-600 rounded text-slate-300 transition-colors"
                            >
                              {copiedExample ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                            </button>
                          </div>
                        )}
                      </div>

                      {/* Compatible Training */}
                      <div className="bg-white rounded-lg border border-slate-200 p-4">
                        <h4 className="font-semibold text-slate-900 mb-3">Compatible Training</h4>
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <div className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2">Methods</div>
                            <div className="flex flex-wrap gap-1">
                              {selectedSample.compatible_methods.map(m => (
                                <span key={m} className={`text-xs px-2 py-1 rounded font-medium ${getMethodColor(m)}`}>
                                  {m.toUpperCase()}
                                </span>
                              ))}
                            </div>
                          </div>
                          {selectedSample.compatible_algorithms.length > 0 && (
                            <div>
                              <div className="text-xs font-medium text-slate-500 uppercase tracking-wide mb-2">Algorithms</div>
                              <div className="flex flex-wrap gap-1">
                                {selectedSample.compatible_algorithms.map(algo => (
                                  <span key={algo} className={`text-xs px-2 py-1 rounded font-medium ${getAlgoColor(algo)}`}>
                                    {algo.toUpperCase()}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Quick Start */}
                      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <h4 className="font-semibold text-blue-900 mb-2 flex items-center gap-2">
                          <AlertCircle className="w-4 h-4" />
                          Quick Start
                        </h4>
                        <ol className="text-sm text-blue-800 space-y-1 list-decimal list-inside">
                          <li>Download the example file using the button above</li>
                          <li>Review the format structure and adapt your data accordingly</li>
                          <li>Upload your dataset in the Dataset Configuration section</li>
                          <li>Select compatible training method and algorithm</li>
                          <li>Start training!</li>
                        </ol>
                      </div>
                    </div>
                  )}

                  {activeTab === 'preview' && (
                    <div className="bg-white rounded-lg border border-slate-200 overflow-hidden">
                      <div className="bg-slate-100 px-4 py-2 border-b border-slate-200">
                        <span className="text-sm font-medium text-slate-700">{selectedSample.filename}</span>
                        <span className="text-xs text-slate-500 ml-2">({selectedSample.preview.length} samples shown)</span>
                      </div>
                      <div className="divide-y divide-slate-200 max-h-96 overflow-y-auto">
                        {selectedSample.preview.map((row, idx) => (
                          <div key={idx} className="p-3">
                            <div className="text-xs text-slate-400 mb-1">Sample {idx + 1}</div>
                            <pre className="text-xs bg-slate-50 p-2 rounded overflow-x-auto whitespace-pre-wrap">
                              {JSON.stringify(row, null, 2)}
                            </pre>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {activeTab === 'docs' && (
                    <div className="bg-white rounded-lg border border-slate-200 p-6">
                      {isLoadingDocs ? (
                        <div className="flex items-center justify-center py-12">
                          <Loader2 className="w-6 h-6 animate-spin text-blue-500 mr-2" />
                          <span className="text-slate-600">Loading documentation...</span>
                        </div>
                      ) : documentation ? (
                        <div className="prose prose-sm prose-slate max-w-none prose-headings:font-bold prose-h1:text-2xl prose-h2:text-xl prose-h2:border-b prose-h2:pb-2 prose-h3:text-lg prose-code:bg-slate-100 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-code:text-sm prose-pre:bg-slate-900 prose-pre:text-slate-100 prose-table:text-sm">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {documentation}
                          </ReactMarkdown>
                        </div>
                      ) : (
                        <div className="text-center py-12 text-slate-500">
                          <BookOpen className="w-12 h-12 mx-auto mb-3 opacity-50" />
                          <p>Documentation not available for this format</p>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              </>
            ) : (
              <div className="flex-1 flex items-center justify-center bg-slate-50">
                <div className="text-center">
                  <FileText className="w-16 h-16 mx-auto mb-4 text-slate-300" />
                  <h3 className="text-lg font-medium text-slate-600 mb-2">Select a Dataset Format</h3>
                  <p className="text-sm text-slate-500">Choose a format from the list to view details and download examples</p>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Footer */}
        <div className="p-3 border-t border-slate-200 bg-slate-50 flex items-center justify-between">
          <div className="text-xs text-slate-500">
            All example datasets are verified to work with USF-BIOS training framework
          </div>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-slate-200 text-slate-700 rounded-lg font-medium hover:bg-slate-300 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  )
}
