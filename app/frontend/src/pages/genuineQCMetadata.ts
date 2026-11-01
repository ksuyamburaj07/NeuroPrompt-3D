import type { QCMetadata } from './CorticalQCSlicePane'

const qcBase = (
  import.meta.env.VITE_QC_MRI_BASE_URL
  || (import.meta.env.DEV ? 'http://127.0.0.1:8013' : '/qc/mri')
)

const expectedSource =
  '4f27af95641365549b4af215aa210bd30a86ff1a00aa7281010daacaa235a2dc'

export async function getGenuineQCMetadata(): Promise<QCMetadata> {
  const response = await fetch(`${qcBase}/metadata`, {
    cache: 'no-store',
  })

  if (!response.ok) {
    throw new Error('Genuine MRI QC service is unavailable.')
  }

  const data = (await response.json()) as QCMetadata

  if (
    data.orientation_convention !== 'RAS+' ||
    data.source_sha256 !== expectedSource ||
    data.reconstruction_id !== 'm11e6b5_trial01' ||
    data.qc_status !== 'provisional' ||
    data.clinically_validated !== false ||
    JSON.stringify(data.shape_ras_xyz) !== '[182,218,182]' ||
    JSON.stringify(data.affine_ras) !==
      '[[1,0,0,-91],[0,1,0,-126],[0,0,1,-72],[0,0,0,1]]' ||
    !data.planes ||
    data.planes.axial?.count !== 182 ||
    data.planes.coronal?.count !== 218 ||
    data.planes.sagittal?.count !== 182
  ) {
    throw new Error('Genuine MRI QC metadata identity mismatch.')
  }

  return data
}
