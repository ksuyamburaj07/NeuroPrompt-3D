import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'

const ALLOWED_RUN = 'run_ee71bd9f95e755f7909f6011d0ef7410'

const ARRAY_NAME =
  /^(baseline_probability|baseline_mask|final_mask|removal_mask|support_mask)_(npy|nifti)$/

function artifactFilename(name) {
  if (name === 'result_json') return 'result.json'

  const match = ARRAY_NAME.exec(name)
  if (!match) return null

  return `${match[1]}.${match[2] === 'npy' ? 'npy' : 'nii.gz'}`
}

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

  const runId = queryValue(req, 'runId')
  const artifactName = queryValue(req, 'artifactName')
  const filename = artifactFilename(artifactName ?? '')

  if (runId !== ALLOWED_RUN || !filename) {
    return res.status(404).json({
      detail: 'Artifact unavailable'
    })
  }

  if (process.env.NP3D_SAVED_RUNS_ENABLED !== 'true') {
    return res.status(503).json({
      detail: 'Artifact service disabled'
    })
  }

  const base = verifiedBase(
    process.env.NP3D_SAVED_RUNS_MODAL_BASE
  )
  const key = process.env.NP3D_MODAL_PROXY_KEY
  const secret = process.env.NP3D_MODAL_PROXY_SECRET

  if (
    !base ||
    !key?.startsWith('wk-') ||
    !secret?.startsWith('ws-')
  ) {
    return res.status(503).json({
      detail: 'Protected artifact service unavailable'
    })
  }

  const path =
    `/api/v1/runs/${runId}/artifacts/${artifactName}`

  try {
    const upstream = await fetch(`${base}${path}`, {
      method: 'GET',
      headers: {
        'Modal-Key': key,
        'Modal-Secret': secret
      },
      redirect: 'manual',
      cache: 'no-store',
      signal: AbortSignal.timeout(55000)
    })

    if (!upstream.ok) {
      await upstream.body?.cancel()
      const status = upstream.status === 404 ? 404
        : upstream.status === 409 ? 409 : 502

      return res.status(status).json({
        detail: 'Artifact could not be retrieved'
      })
    }

    if (!upstream.body) {
      return res.status(502).json({
        detail: 'Artifact stream unavailable'
      })
    }

    res.status(200)
    res.setHeader(
      'Content-Type',
      upstream.headers.get('content-type') ||
        'application/octet-stream'
    )
    res.setHeader(
      'Content-Disposition',
      `attachment; filename="${filename}"`
    )

    // Stream from Modal to the browser without buffering
    // the complete scientific artifact in Vercel.
    await pipeline(Readable.fromWeb(upstream.body), res)

  } catch {
    if (!res.headersSent) {
      return res.status(502).json({
        detail: 'Artifact transfer failed'
      })
    }

    if (!res.destroyed) res.destroy()
  }
}
