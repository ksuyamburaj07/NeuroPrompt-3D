import { gzipSync } from 'node:zlib'

const MAX_COMPRESSED_MESH_BYTES = 4_300_000
const MAX_ORIGINAL_MESH_BYTES = 8_000_000
const MAX_IMAGE_BYTES = 1_000_000

const PNG_SIGNATURE = Buffer.from([
  137, 80, 78, 71, 13, 10, 26, 10
])

const ALLOWED_MESHES = new Set([
  'lh.pial.ras.ply',
  'rh.pial.ras.ply',
  'lh.pial.ras.sulc-qc.ply',
  'rh.pial.ras.sulc-qc.ply'
])

const ALLOWED_PLANES = {
  axial: 181,
  coronal: 217,
  sagittal: 181
}

const ALLOWED_KINDS = new Set([
  'slices',
  'slices3d',
  'contours'
])

function param(req, key) {
  const value = req.query?.[key]
  return typeof value === 'string' ? value : null
}

function validModalBase(value) {
  try {
    const url = new URL(value)

    if (
      url.protocol !== 'https:' ||
      !url.hostname.endsWith('.modal.run') ||
      url.pathname !== '/' ||
      url.search ||
      url.hash ||
      url.username ||
      url.password
    ) {
      return null
    }

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
  res.setHeader('X-Content-Type-Options', 'nosniff')

  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET')
    return res.status(405).end()
  }

  if (process.env.NP3D_QC_PROXY_ENABLED !== 'true') {
    return res.status(503).json({
      detail: 'Protected Cortical QC is disabled'
    })
  }

  const key = process.env.NP3D_MODAL_PROXY_KEY
  const secret = process.env.NP3D_MODAL_PROXY_SECRET

  const base = validModalBase(
    process.env.NP3D_CORTICAL_MODAL_BASE
  )

  if (
    !key?.startsWith('wk-') ||
    !secret?.startsWith('ws-') ||
    !base
  ) {
    return res.status(503).json({
      detail: 'Protected Cortical QC configuration missing'
    })
  }

  const resource = param(req, 'resource')
  let upstreamPath

  if (resource === 'metadata') {
    upstreamPath = '/qc/mri/metadata'

  } else if (resource === 'image') {
    const kind = param(req, 'kind')
    const plane = param(req, 'plane')
    const index = param(req, 'index')

    if (
      !ALLOWED_KINDS.has(kind) ||
      !Object.hasOwn(ALLOWED_PLANES, plane) ||
      !/^[0-9]{1,3}$/.test(index ?? '') ||
      Number(index) > ALLOWED_PLANES[plane]
    ) {
      return res.status(422).json({
        detail: 'Invalid Cortical QC image request'
      })
    }

    upstreamPath = (
      `/qc/mri/${kind}/${plane}/${Number(index)}`
    )

  } else if (resource === 'mesh') {
    const filename = param(req, 'filename')

    if (!ALLOWED_MESHES.has(filename)) {
      return res.status(404).json({
        detail: 'Mesh not available'
      })
    }

    upstreamPath = `/qc/meshes/${filename}`

  } else {
    return res.status(404).json({
      detail: 'Cortical QC resource not found'
    })
  }

  let response

  try {
    response = await fetch(
      `${base}${upstreamPath}`,
      {
        method: 'GET',
        headers: {
          'Modal-Key': key,
          'Modal-Secret': secret
        },
        redirect: 'manual',
        cache: 'no-store',
        signal: AbortSignal.timeout(55000)
      }
    )
  } catch {
    return res.status(502).json({
      detail: 'Protected cortical backend unavailable'
    })
  }

  if (!response.ok) {
    return res.status(502).json({
      detail: 'Verified cortical source unavailable'
    })
  }

  let data

  try {
    data = Buffer.from(await response.arrayBuffer())
  } catch {
    return res.status(502).json({
      detail: 'Could not read verified cortical source'
    })
  }

  if (resource === 'metadata') {
    if (data.length > 100_000) {
      return res.status(502).end()
    }

    try {
      JSON.parse(data.toString('utf8'))
    } catch {
      return res.status(502).end()
    }

    res.setHeader(
      'Content-Type',
      'application/json; charset=utf-8'
    )

    return res.status(200).send(data)
  }

  if (resource === 'image') {
    if (
      data.length < 8 ||
      data.length > MAX_IMAGE_BYTES ||
      !data.subarray(0, 8).equals(PNG_SIGNATURE)
    ) {
      return res.status(502).json({
        detail: 'Invalid original MRI image response'
      })
    }

    res.setHeader('Content-Type', 'image/png')
    return res.status(200).send(data)
  }

  // Original PLY geometry: compress transport only.
  // The browser decodes Content-Encoding: gzip automatically.
  if (
    data.length < 4 ||
    data.length > MAX_ORIGINAL_MESH_BYTES ||
    data.subarray(0, 4).toString('ascii') !== 'ply\n'
  ) {
    return res.status(502).json({
      detail: 'Invalid original cortical mesh'
    })
  }

  let compressed

  try {
    compressed = gzipSync(data, { level: 6 })
  } catch {
    return res.status(502).json({
      detail: 'Cortical mesh compression failed'
    })
  }

  if (compressed.length > MAX_COMPRESSED_MESH_BYTES) {
    return res.status(502).json({
      detail: 'Compressed original mesh exceeds safe limit'
    })
  }

  res.setHeader(
    'Content-Type',
    'application/octet-stream'
  )
  res.setHeader('Content-Encoding', 'gzip')
  res.setHeader('Vary', 'Accept-Encoding')

  return res.status(200).send(compressed)
}
