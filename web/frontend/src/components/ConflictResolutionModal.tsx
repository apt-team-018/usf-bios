'use client'

import { AlertTriangle, Cpu, Zap, Loader2, X, Activity, Layers } from 'lucide-react'

// Types for different conflict scenarios
export type ConflictType = 
  | 'training_while_inference'      // Starting training when inference is loaded
  | 'inference_while_training'      // Loading inference when training is running
  | 'new_inference_replace'         // Loading new model when another is loaded
  | 'training_while_training'       // Starting training when another is running

export type ModelType = 'full' | 'lora' | 'adapter' | 'qlora'

export interface ConflictContext {
  // Current state
  currentModelPath?: string
  currentModelName?: string
  currentAdapterPath?: string
  currentAdapterName?: string
  currentModelType?: ModelType
  currentBackend?: string
  
  // New action
  newModelPath?: string
  newModelName?: string
  newAdapterPath?: string
  newAdapterName?: string
  newModelType?: ModelType
  
  // Training-specific
  trainingJobName?: string
  trainingProgress?: number
  trainingModel?: string
  
  // Inference-specific
  memoryUsedGB?: number
  estimatedMemoryGB?: number
}

export interface ConflictResolutionModalProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: () => void
  conflictType: ConflictType
  context: ConflictContext
  isLoading?: boolean
}

// Generate dynamic messages based on conflict type and context
function getConflictDetails(type: ConflictType, ctx: ConflictContext): {
  title: string
  icon: typeof AlertTriangle
  iconBg: string
  iconColor: string
  message: string
  details: string[]
  warning: string
  confirmText: string
  cancelText: string
  confirmBg: string
} {
  const modelTypeName = (mt?: ModelType) => {
    switch (mt) {
      case 'lora': return 'LoRA Adapter'
      case 'qlora': return 'QLoRA Adapter'
      case 'adapter': return 'Adapter'
      case 'full': return 'Full Model'
      default: return 'Model'
    }
  }

  switch (type) {
    case 'training_while_inference':
      return {
        title: 'Start Training?',
        icon: Zap,
        iconBg: 'bg-amber-100',
        iconColor: 'text-amber-600',
        message: 'You are about to start a new training job, but an inference model is currently loaded in GPU memory.',
        details: [
          `Current Model: ${ctx.currentModelName || ctx.currentModelPath || 'Unknown'}`,
          ctx.currentAdapterPath ? `Active Adapter: ${ctx.currentAdapterName || ctx.currentAdapterPath}` : '',
          ctx.memoryUsedGB ? `GPU Memory Used: ${ctx.memoryUsedGB.toFixed(1)} GB` : '',
          `Backend: ${ctx.currentBackend || 'transformers'}`,
        ].filter(Boolean),
        warning: 'Starting training will automatically unload the current inference model to free GPU memory. This ensures sufficient VRAM is available for the training process.',
        confirmText: 'Stop Inference & Start Training',
        cancelText: 'Keep Inference',
        confirmBg: 'bg-blue-600 hover:bg-blue-700',
      }

    case 'inference_while_training':
      return {
        title: 'Training in Progress',
        icon: Activity,
        iconBg: 'bg-red-100',
        iconColor: 'text-red-600',
        message: 'Cannot load inference model while training is in progress.',
        details: [
          `Training Job: ${ctx.trainingJobName || 'Unknown'}`,
          ctx.trainingModel ? `Model: ${ctx.trainingModel}` : '',
          ctx.trainingProgress !== undefined ? `Progress: ${ctx.trainingProgress.toFixed(1)}%` : '',
        ].filter(Boolean),
        warning: 'Loading an inference model during training would cause GPU memory conflicts and could crash both processes. Please wait for training to complete or stop the training first.',
        confirmText: 'View Training Progress',
        cancelText: 'Close',
        confirmBg: 'bg-blue-600 hover:bg-blue-700',
      }

    case 'new_inference_replace': {
      const isSameBase = ctx.currentModelPath === ctx.newModelPath
      const isAdapterSwitch = isSameBase && ctx.newAdapterPath && ctx.newAdapterPath !== ctx.currentAdapterPath
      const isNewAdapter = !ctx.currentAdapterPath && ctx.newAdapterPath
      const isRemovingAdapter = ctx.currentAdapterPath && !ctx.newAdapterPath
      
      let actionDesc = ''
      if (isAdapterSwitch) {
        actionDesc = `Switch from ${modelTypeName(ctx.currentModelType)} "${ctx.currentAdapterName || 'current'}" to "${ctx.newAdapterName || ctx.newAdapterPath}"`
      } else if (isNewAdapter) {
        actionDesc = `Load ${modelTypeName(ctx.newModelType)} "${ctx.newAdapterName || ctx.newAdapterPath}" on top of the base model`
      } else if (isRemovingAdapter) {
        actionDesc = `Unload the current ${modelTypeName(ctx.currentModelType)} and use base model only`
      } else {
        actionDesc = `Replace current model with "${ctx.newModelName || ctx.newModelPath}"`
      }

      return {
        title: isSameBase ? 'Switch Adapter?' : 'Replace Model?',
        icon: Layers,
        iconBg: 'bg-blue-100',
        iconColor: 'text-blue-600',
        message: actionDesc,
        details: [
          `Current: ${ctx.currentModelName || ctx.currentModelPath || 'Unknown'}`,
          ctx.currentAdapterPath ? `Current Adapter: ${ctx.currentAdapterName || ctx.currentAdapterPath} (${modelTypeName(ctx.currentModelType)})` : 'No adapter loaded',
          '',
          `New: ${ctx.newModelName || ctx.newModelPath || 'Same base model'}`,
          ctx.newAdapterPath ? `New Adapter: ${ctx.newAdapterName || ctx.newAdapterPath} (${modelTypeName(ctx.newModelType)})` : '',
        ].filter(Boolean),
        warning: isSameBase 
          ? 'Switching adapters on the same base model is fast and won\'t require reloading the base model.'
          : 'Loading a different base model will unload the current model and free GPU memory before loading the new one. This may take a few minutes.',
        confirmText: isSameBase ? 'Switch Adapter' : 'Replace Model',
        cancelText: 'Cancel',
        confirmBg: 'bg-blue-600 hover:bg-blue-700',
      }
    }

    case 'training_while_training':
      return {
        title: 'Training Already Running',
        icon: AlertTriangle,
        iconBg: 'bg-red-100',
        iconColor: 'text-red-600',
        message: 'A training job is already in progress. You cannot start a new training until the current one completes.',
        details: [
          `Current Training: ${ctx.trainingJobName || 'Unknown'}`,
          ctx.trainingModel ? `Model: ${ctx.trainingModel}` : '',
          ctx.trainingProgress !== undefined ? `Progress: ${ctx.trainingProgress.toFixed(1)}%` : '',
        ].filter(Boolean),
        warning: 'Only one training job can run at a time to ensure GPU resources are fully available. Please wait for the current training to complete or stop it from the Training History.',
        confirmText: 'View Training Progress',
        cancelText: 'Close',
        confirmBg: 'bg-blue-600 hover:bg-blue-700',
      }

    default:
      return {
        title: 'Confirm Action',
        icon: AlertTriangle,
        iconBg: 'bg-amber-100',
        iconColor: 'text-amber-600',
        message: 'Are you sure you want to proceed?',
        details: [],
        warning: '',
        confirmText: 'Confirm',
        cancelText: 'Cancel',
        confirmBg: 'bg-blue-600 hover:bg-blue-700',
      }
  }
}

export default function ConflictResolutionModal({
  isOpen,
  onClose,
  onConfirm,
  conflictType,
  context,
  isLoading = false,
}: ConflictResolutionModalProps) {
  if (!isOpen) return null

  const details = getConflictDetails(conflictType, context)
  const Icon = details.icon
  
  // For "view training" type actions, confirm just closes and navigates
  const isViewOnly = conflictType === 'inference_while_training' || conflictType === 'training_while_training'

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-[100] p-2 sm:p-4">
      <div 
        className="bg-white rounded-xl sm:rounded-2xl shadow-2xl w-full max-w-[95vw] sm:max-w-lg max-h-[90vh] overflow-y-auto transform transition-all animate-in fade-in zoom-in-95 duration-200"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header - Responsive padding and icon sizing */}
        <div className="flex items-start gap-3 sm:gap-4 p-4 sm:p-6 border-b border-slate-100">
          <div className={`w-10 h-10 sm:w-12 sm:h-12 ${details.iconBg} rounded-lg sm:rounded-xl flex items-center justify-center flex-shrink-0`}>
            <Icon className={`w-5 h-5 sm:w-6 sm:h-6 ${details.iconColor}`} />
          </div>
          <div className="flex-1 min-w-0">
            <h3 className="font-bold text-lg sm:text-xl text-slate-900">{details.title}</h3>
            <p className="text-sm sm:text-base text-slate-600 mt-1">{details.message}</p>
          </div>
          <button 
            onClick={onClose}
            disabled={isLoading}
            className="text-slate-400 hover:text-slate-600 transition-colors disabled:opacity-50 p-1 hover:bg-slate-100 rounded-lg touch-manipulation"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        {/* Details Section - Responsive padding */}
        {details.details.length > 0 && (
          <div className="px-4 sm:px-6 py-3 sm:py-4 bg-slate-50 border-b border-slate-100">
            <div className="space-y-1">
              {details.details.map((detail, idx) => (
                detail ? (
                  <p key={idx} className="text-xs sm:text-sm text-slate-700 flex items-center gap-2 break-words">
                    {detail.startsWith('Current:') || detail.startsWith('New:') ? (
                      <span className="font-semibold">{detail}</span>
                    ) : (
                      <>
                        <span className="w-1.5 h-1.5 bg-slate-400 rounded-full flex-shrink-0" />
                        <span className="break-all">{detail}</span>
                      </>
                    )}
                  </p>
                ) : (
                  <div key={idx} className="h-2" />
                )
              ))}
            </div>
          </div>
        )}
        
        {/* Warning Message - Responsive padding */}
        {details.warning && (
          <div className="px-4 sm:px-6 py-3 sm:py-4">
            <div className={`p-3 sm:p-4 rounded-lg sm:rounded-xl ${
              conflictType === 'inference_while_training' || conflictType === 'training_while_training'
                ? 'bg-red-50 border border-red-100'
                : 'bg-amber-50 border border-amber-100'
            }`}>
              <p className={`text-xs sm:text-sm ${
                conflictType === 'inference_while_training' || conflictType === 'training_while_training'
                  ? 'text-red-700'
                  : 'text-amber-700'
              }`}>
                {details.warning}
              </p>
            </div>
          </div>
        )}
        
        {/* Footer with Actions - Responsive with stacked buttons on mobile */}
        <div className="flex flex-col sm:flex-row gap-2 sm:gap-3 px-4 sm:px-6 pb-4 sm:pb-6 pt-2">
          <button
            onClick={onClose}
            disabled={isLoading}
            className="flex-1 px-4 py-2.5 sm:py-3 border border-slate-300 text-slate-700 rounded-lg sm:rounded-xl font-medium hover:bg-slate-50 transition-colors disabled:opacity-50 touch-manipulation min-h-[44px]"
          >
            {details.cancelText}
          </button>
          <button
            onClick={onConfirm}
            disabled={isLoading}
            className={`flex-1 px-4 py-2.5 sm:py-3 ${details.confirmBg} text-white rounded-lg sm:rounded-xl font-semibold transition-colors disabled:opacity-50 flex items-center justify-center gap-2 touch-manipulation min-h-[44px]`}
          >
            {isLoading && (
              <Loader2 className="w-4 h-4 animate-spin" />
            )}
            <span className="text-sm sm:text-base">{details.confirmText}</span>
          </button>
        </div>
      </div>
    </div>
  )
}

// Helper hook for managing conflict resolution state
export interface ConflictState {
  isOpen: boolean
  conflictType: ConflictType | null
  context: ConflictContext
  onResolve: (() => void) | null
}

export const initialConflictState: ConflictState = {
  isOpen: false,
  conflictType: null,
  context: {},
  onResolve: null,
}
