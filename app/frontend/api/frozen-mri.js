const MODAL_BASE =
  'https://ksuyamburaj07--neuroprompt3d-real-backend-private-mri-api.us-east.modal.run'

const MODALITIES = new Set(['t1n', 't1c', 't2w', 't2f'])
const LAYERS = new Set([
  'none', 'baseline', 'final', 'removed', 'support'
])
const PLANES = new Set(['axial', 'coronal', 'sagittal'])

const PNG_SIGNATURE = Buffer.from([
  137, 80, 78, 71, 13, 10, 26, 10
])

export default async function handler(req, res) {
  res.setHeader('Cache-Control', 'private, no-store, max-age=0')
  res.setHeader('X-Content-Type-Options', 'nosniff')

  if (req.method !== 'GET') {
    res.setHeader('Allow', 'GET')
    return res.status(405).end()
  }

  if (process.env.NP3D_MRI_PROXY_ENABLED !== 'true') {
    return res.status(503).json({
      detail: 'Protected MRI proxy not enabled'
    })
  }

  const key = process.env.NP3D_MODAL_PROXY_KEY
  const secret = process.env.NP3D_MODAL_PROXY_SECRET

  if (!key?.startsWith('wk-') || !secret?.startsWith('ws-')) {
    return res.status(503).json({
      detail: 'MRI proxy configuration unavailable'
    })
  }

  const get = (name) =>
    typeof req.query?.[name] === 'string'
      ? req.query[name]
      : null

  const caseIndex = Number(get('caseIndex'))
  const plane = get('plane')
  const modality = get('modality') ?? 't2f'
  const layer = get('layer') ?? 'removed'
  const contrast = get('contrast') ?? 'slice'

  if (
    !Number.isInteger(caseIndex) ||
    caseIndex < 1 ||
    caseIndex > 129 ||
    !PLANES.has(plane) ||
    !MODALITIES.has(modality) ||
    !LAYERS.has(layer) ||
    !['slice', 'volume'].includes(contrast)
  ) {
    return res.status(422).json({
      detail: 'Invalid MRI request'
    })
  }

  const axis = plane === 'axial' ? 'z' : 'index'
  const index = Number(get(axis))
  const maximum = plane === 'coronal' ? 217 : 181

  if (
    !Number.isInteger(index) ||
    index < 0 ||
    index > maximum
  ) {
    return res.status(422).json({
      detail: 'Invalid MRI coordinate'
    })
  }

  const params = new URLSearchParams({
    [axis]: String(index),
    modality,
    layer,
    contrast
  })

  const url =
    `${MODAL_BASE}/api/v1/frozen-test/cases/` +
    `${caseIndex}/slices/${plane}?${params}`

  try {
    const upstream = await fetch(url, {
      method: 'GET',
      headers: {
        'Modal-Key': key,
        'Modal-Secret': secret
      },
      redirect: 'manual',
      cache: 'no-store',
      signal: AbortSignal.timeout(45000)
    })

    if (!upstream.ok) {
      return res.status(upstream.status === 422 ? 422 : 502)
        .json({ detail: 'Verified MRI unavailable' })
    }

    const image = Buffer.from(await upstream.arrayBuffer())

    if (
      image.length < 8 ||
      image.length > 1000000 ||
      !image.subarray(0, 8).equals(PNG_SIGNATURE)
    ) {
      return res.status(502).json({
        detail: 'Invalid verified image response'
      })
    }

    res.setHeader('Content-Type', 'image/png')
    return res.status(200).send(image)

  } catch {
    return res.status(502).json({
      detail: 'MRI service temporarily unavailable'
    })
  }
}
