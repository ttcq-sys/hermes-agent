const MESSAGING_SETTINGS_PATH = /^\/api\/messaging\/platforms(?:\/|$)/

export function requireMessagingRequestProfile(path: unknown, profile: unknown): string | undefined {
  const normalizedProfile = typeof profile === 'string' ? profile.trim() : ''

  if (typeof path === 'string' && MESSAGING_SETTINGS_PATH.test(path.split(/[?#]/, 1)[0]) && !normalizedProfile) {
    throw new Error('Messaging settings requests require an explicit profile; refusing to use the primary profile.')
  }

  return normalizedProfile || undefined
}
