const RUN_PATTERN = /^run_[0-9a-f]{32}$/

function queryValue(req, name) {
  const value = req.query?.[name]
  return typeof value === 'string' ? value : null
}

function verifiedBase(raw) {
  try {
    const url = new URL(raw)

    if (
      url.protocol !== 'https:' ||
      !url.hostname.endsWith('.modal.run') ||
      url.pathname !== '/' ||
      url.search ||
      url.hash ||
      url.username ||
      url.password
    ) return null

    return url.origin
  } catch {
    return null
  }
}

export default async function handler(req, res) {
  res.setHeader(
    'Cache-Control',
    'private, no-store, max-age=0'
  )
  res.setHeader(
    'X-Content-Type-Options',
    'nosniff'
  )

  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET')
    return res.status(405).end()
  }

  if (
    process.env.NP3D_SAVED_RUNS_ENABLED !== 'true'
  ) {
    return res.status(503).json({
      detail: 'Saved-run inspection disabled'
    })
  }

  const key = process.env.NP3D_MODAL_PROXY_KEY
  const secret = process.env.NP3D_MODAL_PROXY_SECRET
  const base = verifiedBase(
    process.env.NP3D_SAVED_RUNS_MODAL_BASE
  )

  if (
    !key?.startsWith('wk-') ||
    !secret?.startsWith('ws-') ||
    !base
  ) {
    return res.status(503).json({
      detail: 'Protected run service unavailable'
    })
  }

  const resource = queryValue(req, 'resource')
  let path

  if (resource === 'list') {
    const rawLimit = queryValue(req, 'limit') ?? '20'

    if (!/^[0-9]{1,2}$/.test(rawLimit)) {
      return res.status(422).end()
    }

    const limit = Number(rawLimit)

    if (limit < 1 || limit > 50) {
      return res.status(422).end()
    }

    path = `/api/v1/research/runs?limit=${limit}`

  } else if (resource === 'run') {
    const runId = queryValue(req, 'runId')

    if (!RUN_PATTERN.test(runId ?? '')) {
      return res.status(422).end()
    }

    path = `/api/v1/runs/${runId}`

  } else {
    return res.status(404).end()
  }

  try {
    const upstream = await fetch(
      `${base}${path}`,
      {
        headers: {
          'Modal-Key': key,
          'Modal-Secret': secret
        },
        redirect: 'manual',
        cache: 'no-store',
        signal: AbortSignal.timeout(45000)
      }
    )

    if (upstream.status === 404) {
      return res.status(404).json({
        detail: 'Saved run not found'
      })
    }

    if (!upstream.ok) {
      return res.status(502).json({
        detail: 'Saved-run backend unavailable'
      })
    }

    const data = Buffer.from(
      await upstream.arrayBuffer()
    )

    if (data.length > 100_000) {
      return res.status(502).end()
    }

    const payload = JSON.parse(
      data.toString('utf8')
    )

    if (
      typeof payload !== 'object' ||
      payload === null
    ) {
      return res.status(502).end()
    }

    res.setHeader(
      'Content-Type',
      'application/json; charset=utf-8'
    )

    return res.status(200).json(payload)

  } catch {
    return res.status(502).json({
      detail: 'Saved-run service unreachable'
    })
  }
}
