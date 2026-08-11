import { describe, expect, it } from 'vitest'

import { requireMessagingRequestProfile } from './messaging-request-routing'

describe('desktop messaging request routing', () => {
  it('rejects a messaging settings request without an explicit profile', () => {
    expect(() => requireMessagingRequestProfile('/api/messaging/platforms/telegram', undefined)).toThrow(
      /explicit profile/i
    )
  })

  it('keeps an explicit messaging profile as the backend route target', () => {
    expect(requireMessagingRequestProfile('/api/messaging/platforms/telegram', ' front-source-steward ')).toBe(
      'front-source-steward'
    )
  })

  it('preserves legacy implicit routing for unrelated desktop APIs', () => {
    expect(requireMessagingRequestProfile('/api/status', undefined)).toBeUndefined()
  })
})
