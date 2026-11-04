const CASE = /^case_[0-9a-f]{32}$/
const RUN = /^run_[0-9a-f]{32}$/
const UUID4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i

function value(req, name) {
  const item = req.query?.[name]
  return typeof item === 'string' ? item : null
}

function verifiedBase(raw) {
  try {
    const url = new URL(raw)
    if (
      url.protocol !== 'https:' ||
      !url.hostname.endsWith('.modal.run') ||
      url.pathname !== '/' ||
      url.search || url.hash ||
      url.username || url.password
    ) return null
    return url.origin
  } catch {
    return null
  }
}

export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'private, no-store')
  res.setHeader('X-Content-Type-Options', 'nosniff')

  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST')
    return res.status(405).end()
  }

  if (process.env.NP3D_RUN_CONTROL_ENABLED !== 'true') {
    return res.status(503).json({
      detail: 'Scientific run control disabled'
    })
  }

  const base = verifiedBase(
    process.env.NP3D_RUN_CONTROL_MODAL_BASE
  )
  const key = process.env.NP3D_MODAL_PROXY_KEY
  const secret = process.env.NP3D_MODAL_PROXY_SECRET

  if (
    !base ||
    !key?.startsWith('wk-') ||
    !secret?.startsWith('ws-')
  ) {
    return res.status(503).json({
      detail: 'Protected controller unavailable'
    })
  }

  const operation = value(req, 'operation')
  const caseId = value(req, 'caseId')
  const runId = value(req, 'runId')

  let path
  const headers = {
    'Modal-Key': key,
    'Modal-Secret': secret,
  }

  if (operation === 'create' && CASE.test(caseId ?? '')) {
    const requestKey = req.headers['idempotency-key']

    if (
      typeof requestKey !== 'string' ||
      !UUID4.test(requestKey)
    ) {
      return res.status(422).json({
        detail: 'Valid UUID v4 request key required'
      })
    }

    headers['Idempotency-Key'] = requestKey
    path = `/api/v1/cases/${caseId}/runs`

  } else if (
    operation === 'execute' &&
    RUN.test(runId ?? '')
  ) {
    path = `/api/v1/runs/${runId}/execute`

  } else {
    return res.status(404).json({
      detail: 'Run operation unavailable'
    })
  }

  try {
    const upstream = await fetch(`${base}${path}`, {
      method: 'POST',
      headers,
      redirect: 'manual',
      cache: 'no-store',
      signal: AbortSignal.timeout(55000),
    })

    const data = Buffer.from(await upstream.arrayBuffer())

    if (data.length > 100000) {
      return res.status(502).end()
    }

    let payload
    try {
      payload = JSON.parse(data.toString('utf8'))
    } catch {
      return res.status(502).json({
        detail: 'Invalid controller response'
      })
    }

    const allowed = new Set([
      400, 403, 404, 409, 422, 503
    ])

    const status = upstream.ok
      ? upstream.status
      : allowed.has(upstream.status) ? upstream.status : 502

    return res.status(status).json(payload)

  } catch {
    return res.status(502).json({
      detail: 'Run-controller request could not be confirmed'
    })
  }
}
