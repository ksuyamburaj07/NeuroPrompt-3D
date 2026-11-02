import { gzipSync } from 'node:zlib'

const CASE = /^case_[0-9a-f]{32}$/
const RUN = 'run_[0-9a-f]{32}'
const PLANE = '(?:axial|coronal|sagittal)'
const LAYER = '(?:baseline|final|removed)'
const INDEX = '[0-9]{1,3}'

const ALLOWED = [
  /^metadata$/,
  new RegExp(
    `^slices/(?:t1n|t1c|t2w|t2f)/${PLANE}/${INDEX}$`
  ),
  new RegExp(
    `^overlays/${RUN}/${LAYER}/${PLANE}/${INDEX}$`
  ),
  new RegExp(`^markers/${RUN}$`),
  new RegExp(`^meshes/${RUN}/${LAYER}$`),
  /^anatomy\/brain$/,
  /^cortex\/status$/,
  /^cortex\/(?:lh|rh)$/,
  new RegExp(`^cortex/contours/${PLANE}/${INDEX}$`)
]

const PNG = Buffer.from([137,80,78,71,13,10,26,10])
const MAX_COMPRESSED = 4_300_000

function param(req, key) {
  const value = req.query?.[key]
  if (typeof value === 'string') return value
  if (key === 'path' && Array.isArray(value)) {
    return value.join('/')
  }
  return null
}

function validOrigin(raw) {
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

  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET')
    return res.status(405).end()
  }

  if (process.env.NP3D_LIVE_VIEWER_ENABLED !== 'true') {
    return res.status(503).json({
      detail: 'Live viewer disabled'
    })
  }

  const base = validOrigin(
    process.env.NP3D_LIVE_VIEWER_MODAL_BASE
  )
  const key = process.env.NP3D_MODAL_PROXY_KEY
  const secret = process.env.NP3D_MODAL_PROXY_SECRET

  if (
    !base ||
    !key?.startsWith('wk-') ||
    !secret?.startsWith('ws-')
  ) {
    return res.status(503).json({
      detail: 'Live viewer configuration unavailable'
    })
  }

  const caseId = param(req, 'caseId')
  const path = param(req, 'path')

  if (
    !CASE.test(caseId ?? '') ||
    !path ||
    !ALLOWED.some(pattern => pattern.test(path))
  ) {
    return res.status(404).end()
  }

  const allowedQueries = new Set([
    'caseId', 'path',
    'transparent_background', 'step', 'finish'
  ])

  for (const name of Object.keys(req.query ?? {})) {
    if (!allowedQueries.has(name)) {
      return res.status(422).end()
    }
  }

  const params = new URLSearchParams()

  for (const name of [
    'transparent_background', 'step', 'finish'
  ]) {
    const value = param(req, name)
    if (value !== null) params.set(name, value)
  }

  const query = params.toString()
  const url = (
    `${base}/api/v1/cases/${caseId}/viewer/${path}` +
    (query ? `?${query}` : '')
  )

  try {
    const upstream = await fetch(url, {
      headers: {
        'Modal-Key': key,
        'Modal-Secret': secret
      },
      redirect: 'manual',
      cache: 'no-store',
      signal: AbortSignal.timeout(55000)
    })

    if (!upstream.ok) {
      const status =
        [400, 404, 409, 422].includes(upstream.status)
          ? upstream.status : 502

      return res.status(status).json({
        detail: 'Verified live viewer resource unavailable'
      })
    }

    const bytes = Buffer.from(
      await upstream.arrayBuffer()
    )
    const contentType = upstream.headers.get(
      'content-type'
    ) ?? ''

    if (contentType.includes('application/json')) {
      if (bytes.length > 100_000) {
        return res.status(502).end()
      }

      const payload = JSON.parse(
        bytes.toString('utf8')
      )

      return res.status(200).json(payload)
    }

    if (contentType.includes('image/png')) {
      if (
        bytes.length > 1_000_000 ||
        !bytes.subarray(0, 8).equals(PNG)
      ) {
        return res.status(502).end()
      }

      res.setHeader('Content-Type', 'image/png')
      return res.status(200).send(bytes)
    }

    if (
      bytes.length > 30_000_000 ||
      bytes.subarray(0, 4).toString('ascii') !== 'ply\n'
    ) {
      return res.status(502).end()
    }

    const compressed = gzipSync(bytes, { level: 6 })

    if (compressed.length > MAX_COMPRESSED) {
      return res.status(413).json({
        detail: 'Original mesh exceeds proxy transfer limit'
      })
    }

    res.setHeader(
      'Content-Type', 'application/octet-stream'
    )
    res.setHeader('Content-Encoding', 'gzip')
    res.setHeader('Vary', 'Accept-Encoding')

    return res.status(200).send(compressed)

  } catch {
    return res.status(502).json({
      detail: 'Protected live viewer unavailable'
    })
  }
}
