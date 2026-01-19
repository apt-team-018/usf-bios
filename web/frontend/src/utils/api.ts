/**
 * API URL Detection Utility
 * 
 * Reads backend URL from /runtime-config.json which is written by the
 * entrypoint script at container startup. This is the most reliable method
 * as the backend knows its own external URL from environment variables.
 * 
 * Fallback: hostname-based detection for cases where config isn't available.
 */

// Cached API URL - loaded once at startup, used for entire runtime
let _cachedApiUrl: string | null = null
let _configPromise: Promise<string> | null = null

interface RuntimeConfig {
  backendUrl: string
  frontendUrl: string
  timestamp: string
}

/**
 * Get fallback backend URL based on current hostname.
 * Used only when runtime-config.json is not available.
 */
function getFallbackBackendUrl(): string {
  if (typeof window === 'undefined') return ''
  
  const hostname = window.location.hostname
  const protocol = window.location.protocol
  
  // RunPod, Vast.ai, etc. - port is in hostname
  if (hostname.includes('-3000')) {
    return `${protocol}//${hostname.replace(/-3000/g, '-8000')}`
  }
  
  // Local development
  if (hostname === 'localhost' || hostname === '127.0.0.1') {
    return 'http://localhost:8000'
  }
  
  // Default: same hostname, port 8000
  return `${protocol}//${hostname}:8000`
}

/**
 * Load backend URL from runtime config file.
 * This file is written by entrypoint.sh at container startup.
 */
async function loadRuntimeConfig(): Promise<string> {
  try {
    console.log('[API] Loading runtime config from /runtime-config.json...')
    
    const response = await fetch('/runtime-config.json', {
      cache: 'no-store', // Always get fresh config
    })
    
    if (!response.ok) {
      throw new Error(`Config not found: ${response.status}`)
    }
    
    const config: RuntimeConfig = await response.json()
    console.log('[API] Runtime config loaded:', config)
    console.log('[API] Backend URL:', config.backendUrl)
    
    return config.backendUrl
  } catch (error) {
    console.warn('[API] Failed to load runtime config, using fallback:', error)
    const fallback = getFallbackBackendUrl()
    console.log('[API] Fallback backend URL:', fallback)
    return fallback
  }
}

/**
 * Initialize API URL. Call this once at app startup.
 * Loads from runtime-config.json written by entrypoint script.
 */
export async function initApiUrl(): Promise<string> {
  if (_cachedApiUrl !== null) {
    return _cachedApiUrl
  }
  
  // Prevent multiple simultaneous loads
  if (_configPromise === null) {
    _configPromise = loadRuntimeConfig().then(url => {
      _cachedApiUrl = url
      return url
    })
  }
  
  return _configPromise
}

/**
 * Get the cached API URL synchronously.
 * Returns fallback if not yet loaded.
 */
export function getApiUrl(): string {
  if (_cachedApiUrl !== null) {
    return _cachedApiUrl
  }
  
  // Start loading in background if not already started
  if (_configPromise === null && typeof window !== 'undefined') {
    initApiUrl()
  }
  
  // Return fallback synchronously while config loads
  return getFallbackBackendUrl()
}

/**
 * Pre-computed API URL for use in components.
 * This starts detection and returns best guess synchronously.
 */
export const API_URL = typeof window !== 'undefined' ? getApiUrl() : ''

/**
 * Helper to make API calls with the correct base URL.
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
