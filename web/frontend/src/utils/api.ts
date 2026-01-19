/**
 * API URL Configuration
 * 
 * Reads backend URL from /runtime-config.json ONCE at startup.
 * This file is written by entrypoint.sh which knows the external URL.
 * The URL is cached permanently - no re-reading after startup.
 */

// PERMANENT cache - set once at startup, never changes
let _backendUrl: string | null = null
let _initPromise: Promise<string> | null = null

/**
 * Fallback URL based on hostname (only used if config file missing)
 */
function getFallbackUrl(): string {
  if (typeof window === 'undefined') return ''
  const h = window.location.hostname
  const p = window.location.protocol
  if (h.includes('-3000')) return `${p}//${h.replace(/-3000/g, '-8000')}`
  if (h === 'localhost' || h === '127.0.0.1') return 'http://localhost:8000'
  return `${p}//${h}:8000`
}

/**
 * Initialize: Read config file ONCE, cache forever.
 * Called automatically on first API call.
 */
export async function initApiUrl(): Promise<string> {
  // Already initialized - return cached value
  if (_backendUrl !== null) return _backendUrl
  
  // Already initializing - wait for it
  if (_initPromise !== null) return _initPromise
  
  // First call - read config ONCE
  _initPromise = (async (): Promise<string> => {
    try {
      console.log('[API] Reading config (ONE TIME ONLY)...')
      const res = await fetch('/runtime-config.json')
      if (!res.ok) throw new Error('Config not found')
      const cfg = await res.json()
      _backendUrl = cfg.backendUrl || getFallbackUrl()
      console.log('[API] Backend URL (cached permanently):', _backendUrl)
    } catch {
      _backendUrl = getFallbackUrl()
      console.log('[API] Using fallback (cached permanently):', _backendUrl)
    }
    return _backendUrl as string
  })()
  
  return _initPromise
}

/**
 * Get cached backend URL. Returns fallback if not yet initialized.
 */
export function getApiUrl(): string {
  if (_backendUrl !== null) return _backendUrl
  if (typeof window !== 'undefined') initApiUrl() // Start loading
  return getFallbackUrl()
}

/** Cached API URL for components */
export const API_URL = typeof window !== 'undefined' ? getApiUrl() : ''

/**
 * Make API call - waits for URL init on first call.
 * Waits for URL detection to complete on first call.
 */
export async function apiFetch(endpoint: string, options?: RequestInit): Promise<Response> {
  const baseUrl = await initApiUrl()
  return fetch(`${baseUrl}${endpoint}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
  })
}

/**
 * Helper to make API calls and parse JSON response.
 */
export async function apiJson<T = unknown>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await apiFetch(endpoint, options)
  if (!response.ok) {
    throw new Error(`API error: ${response.status} ${response.statusText}`)
  }
  return response.json()
}
