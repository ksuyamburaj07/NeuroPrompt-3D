const SESSION = /^up_[0-9a-f]{32}$/
const FIELDS = new Set([
  't1n', 't1c', 't2w', 't2f', 'segmentation'
])
const CHUNK_LIMIT = 768 * 1024

function value(req, name) {
  const result = req.query?.[name]
  return typeof result === 'string' ? result : null
}

export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'private, no-store')
  res.setHeader('X-Content-Type-Options', 'nosniff')

  if (process.env.NP3D_UPLOAD_ENABLED !== 'true') {
    return res.status(503).json({
      detail: 'MRI uploads are disabled'
    })
  }

  const key = process.env.NP3D_MODAL_PROXY_KEY
  const secret = process.env.NP3D_MODAL_PROXY_SECRET
  const rawBase = process.env.NP3D_UPLOAD_MODAL_BASE

  let base

  try {
    const url = new URL(rawBase)

    if (
      url.protocol !== 'https:' ||
      !url.hostname.endsWith('.modal.run') ||
      url.pathname !== '/' ||
      url.search ||
      url.hash ||
      url.username ||
      url.password
    ) throw new Error('Invalid backend')

    base = url.origin
  } catch {
    return res.status(503).json({
      detail: 'MRI upload backend unconfigured'
    })
  }

  if (
    !key?.startsWith('wk-') ||
    !secret?.startsWith('ws-')
  ) {
    return res.status(503).json({
      detail: 'MRI upload credentials unavailable'
    })
  }

  const operation = value(req, 'operation')
  const sessionId = value(req, 'sessionId')
  const field = value(req, 'field')

  let path
  let method

  if (operation === 'create' && req.method === 'POST') {
    path = '/api/v1/upload-sessions'
    method = 'POST'
  } else if (
    SESSION.test(sessionId ?? '') &&
    operation === 'status' &&
    req.method === 'GET'
  ) {
    path = `/api/v1/upload-sessions/${sessionId}`
    method = 'GET'
  } else if (
    SESSION.test(sessionId ?? '') &&
    operation === 'chunk' &&
    FIELDS.has(field) &&
    req.method === 'PUT'
  ) {
    const offset = value(req, 'offset')

    if (!/^(0|[1-9][0-9]{0,8})$/.test(offset ?? '')) {
      return res.status(422).end()
    }

    path = (
      `/api/v1/upload-sessions/${sessionId}` +
      `/files/${field}?offset=${offset}`
    )
    method = 'PUT'
  } else if (
    SESSION.test(sessionId ?? '') &&
    operation === 'finalize' &&
    req.method === 'POST'
  ) {
    path = `/api/v1/upload-sessions/${sessionId}/finalize`
    method = 'POST'
  } else if (
    operation === 'discard' &&
    req.method === 'DELETE' &&
    /^case_[0-9a-f]{32}$/.test(value(req, 'caseId') ?? '')
  ) {
    path = `/api/v1/cases/${value(req, 'caseId')}`
    method = 'DELETE'
  } else {
    return res.status(404).json({
      detail: 'Upload operation unavailable'
    })
  }

  const headers = {
    'Modal-Key': key,
    'Modal-Secret': secret
  }

  const sessionToken = req.headers['x-np3d-session-token']

  if (method !== 'POST' || operation !== 'create') {
    if (
      typeof sessionToken !== 'string' ||
      sessionToken.length > 256
    ) return res.status(401).end()

    headers['X-NP3D-Session-Token'] = sessionToken

    if (operation === 'discard') {
      const uploadId = req.headers['x-np3d-upload-id']

      if (
        typeof uploadId !== 'string' ||
        !SESSION.test(uploadId)
      ) return res.status(401).end()

      headers['X-NP3D-Upload-Id'] = uploadId
    }
  }

  let body

  if (operation === 'create') {
    const input = req.body

    if (!input || typeof input !== 'object') {
      return res.status(400).end()
    }

    body = Buffer.from(JSON.stringify(input))

    if (body.length > 12000) {
      return res.status(413).end()
    }

    headers['Content-Type'] = 'application/json'
  } else if (operation === 'chunk') {
    if (!Buffer.isBuffer(req.body)) {
      return res.status(415).json({
        detail: 'Binary chunk expected'
      })
    }

    body = req.body

    if (body.length < 1 || body.length > CHUNK_LIMIT) {
      return res.status(413).end()
    }

    headers['Content-Type'] = 'application/octet-stream'
  }

  try {
    const upstream = await fetch(`${base}${path}`, {
      method,
      headers,
      body,
      redirect: 'manual',
      cache: 'no-store',
      signal: AbortSignal.timeout(55000)
    })

    const content = await upstream.arrayBuffer()

    if (content.byteLength > 100000) {
      return res.status(502).end()
    }

    let payload

    try {
      payload = JSON.parse(
        Buffer.from(content).toString('utf8')
      )
    } catch {
      return res.status(502).json({
        detail: 'Unreadable upload response'
      })
    }

    const allowedErrors = new Set([
      400, 401, 403, 404, 409, 410, 413, 422, 429, 503
    ])

    const status = upstream.ok
      ? upstream.status
      : allowedErrors.has(upstream.status)
        ? upstream.status
        : 502

    return res.status(status).json(payload)

  } catch {
    return res.status(502).json({
      detail: 'MRI upload backend unreachable'
    })
  }
}
