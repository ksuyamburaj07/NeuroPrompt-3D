/**
 * Read-only frontend access to verified M9E6 results.
 * No inference, mutation, or fallback metrics.
 */

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV ? 'http://127.0.0.1:8000/api/v1' : '/api/v1')
).replace(/\/+$/, '')

export type FrozenMetricSummary = {
  baseline_dice_mean: number
  final_dice_mean: number
  delta_dice_mean: number
  baseline_iou_mean: number
  final_iou_mean: number
  delta_iou_mean: number
  baseline_hd95_mm_mean: number
  final_hd95_mm_mean: number
  delta_hd95_mm_mean: number
}

export type FrozenTestSummary = {
  milestone: string
  status: string
  case_count: number
  evaluation_timestamp_utc: string
  evaluation_sha256: string
  overall_summary: FrozenMetricSummary
  action_counts: {
    APPLY_LOCAL_FP: number
    ABSTAIN_BASELINE: number
  }
}

function isRecord(
  value: unknown,
): value is Record<string, unknown> {
  return (
    typeof value === 'object' &&
    value !== null &&
    !Array.isArray(value)
  )
}

function isFrozenTestSummary(
  value: unknown,
): value is FrozenTestSummary {
  if (!isRecord(value)) return false

  const metrics = value.overall_summary
  const actions = value.action_counts

  if (!isRecord(metrics) || !isRecord(actions)) {
    return false
  }

  const requiredMetrics = [
    'baseline_dice_mean',
    'final_dice_mean',
    'delta_dice_mean',
    'baseline_iou_mean',
    'final_iou_mean',
    'delta_iou_mean',
    'baseline_hd95_mm_mean',
    'final_hd95_mm_mean',
    'delta_hd95_mm_mean',
  ]

  const metricsValid = requiredMetrics.every(
    (key) =>
      typeof metrics[key] === 'number' &&
      Number.isFinite(metrics[key]),
  )

  const applied = actions.APPLY_LOCAL_FP
  const abstained = actions.ABSTAIN_BASELINE

  return (
    value.milestone === 'M9E6' &&
    value.status === 'final_test_evaluation' &&
    value.case_count === 129 &&
    typeof value.evaluation_timestamp_utc === 'string' &&
    typeof value.evaluation_sha256 === 'string' &&
    /^[a-f0-9]{64}$/.test(value.evaluation_sha256) &&
    metricsValid &&
    typeof applied === 'number' &&
    Number.isInteger(applied) &&
    typeof abstained === 'number' &&
    Number.isInteger(abstained) &&
    applied === 44 &&
    abstained === 85
  )
}

export async function getFrozenTestSummary(
  signal?: AbortSignal,
): Promise<FrozenTestSummary> {
  const response = await fetch(
    `${API_BASE}/frozen-test/summary`,
    {
      method: 'GET',
      cache: 'no-store',
      signal,
    },
  )

  if (!response.ok) {
    throw new Error(
      `Verified frozen evidence unavailable (HTTP ${response.status}).`,
    )
  }

  const payload: unknown = await response.json()

  if (!isFrozenTestSummary(payload)) {
    throw new Error(
      'Frozen evaluation response failed validation.',
    )
  }

  return payload
}


export type FrozenAction =
  | 'APPLY_LOCAL_FP'
  | 'ABSTAIN_BASELINE'

export type FrozenCase = {
  case_index: number
  action: FrozenAction
  gate_state: string
  sam_used: boolean
  hotspot_variance: number
  baseline_dice: number
  final_dice: number
  delta_dice: number
  baseline_iou: number
  final_iou: number
  delta_iou: number
  baseline_hd95_mm: number
  final_hd95_mm: number
  delta_hd95_mm: number
}

export type FrozenCaseList = {
  milestone: string
  total_cases: number
  total_matching: number
  offset: number
  limit: number
  action_filter: FrozenAction | null
  cases: FrozenCase[]
}

export type FrozenCaseDetail = {
  milestone: string
  evaluation_sha256: string
  case: FrozenCase
}

const FROZEN_ACTIONS = [
  'APPLY_LOCAL_FP',
  'ABSTAIN_BASELINE',
]

function isFrozenAction(value: unknown): value is FrozenAction {
  return typeof value === 'string' &&
    FROZEN_ACTIONS.includes(value)
}

function isFrozenCase(value: unknown): value is FrozenCase {
  if (!isRecord(value)) return false

  if ('case_id' in value || 'ground_truth_voxels' in value) {
    return false
  }

  const numericFields = [
    'hotspot_variance',
    'baseline_dice',
    'final_dice',
    'delta_dice',
    'baseline_iou',
    'final_iou',
    'delta_iou',
    'baseline_hd95_mm',
    'final_hd95_mm',
    'delta_hd95_mm',
  ]

  return (
    typeof value.case_index === 'number' &&
    Number.isInteger(value.case_index) &&
    value.case_index >= 1 &&
    value.case_index <= 129 &&
    isFrozenAction(value.action) &&
    (
      (value.action === 'APPLY_LOCAL_FP' &&
        value.gate_state === 'FP_ELIGIBLE' &&
        value.sam_used === true) ||
      (value.action === 'ABSTAIN_BASELINE' &&
        value.gate_state === 'ABSTAIN' &&
        value.sam_used === false)
    ) &&
    numericFields.every(
      key => typeof value[key] === 'number' &&
        Number.isFinite(value[key]),
    )
  )
}

async function getVerifiedCaseResponse(
  path: string,
  signal?: AbortSignal,
): Promise<unknown> {
  let response: Response

  try {
    response = await fetch(`${API_BASE}/${path}`, {
      method: 'GET',
      cache: 'no-store',
      signal,
    })
  } catch (error) {
    if (signal?.aborted) throw error
    throw new Error('Cannot reach the frozen evidence API.', { cause: error })
  }

  if (!response.ok) {
    throw new Error(
      `Frozen evidence unavailable (HTTP ${response.status}).`,
    )
  }

  try {
    return await response.json() as unknown
  } catch {
    throw new Error('Frozen evidence API returned invalid JSON.')
  }
}

export async function getFrozenTestCases(
  offset = 0,
  limit = 20,
  action: FrozenAction | null = null,
  signal?: AbortSignal,
): Promise<FrozenCaseList> {
  const params = new URLSearchParams({
    offset: String(offset),
    limit: String(limit),
  })

  if (action !== null) params.set('action', action)

  const payload = await getVerifiedCaseResponse(
    `frozen-test/cases?${params.toString()}`,
    signal,
  )

  if (!isRecord(payload) || !Array.isArray(payload.cases)) {
    throw new Error('Invalid frozen case-list response.')
  }

  if (
    payload.milestone !== 'M9E6' ||
    payload.total_cases !== 129 ||
    typeof payload.total_matching !== 'number' ||
    !Number.isInteger(payload.total_matching) ||
    payload.total_matching < 0 ||
    payload.total_matching > 129 ||
    payload.offset !== offset ||
    payload.limit !== limit ||
    payload.action_filter !== action ||
    payload.cases.length > limit ||
    !payload.cases.every(isFrozenCase) ||
    !payload.cases.every(
      (row: FrozenCase) => action === null || row.action === action,
    )
  ) {
    throw new Error('Frozen case-list validation failed.')
  }

  return payload as FrozenCaseList
}

export async function getFrozenTestCase(
  index: number,
  signal?: AbortSignal,
): Promise<FrozenCaseDetail> {
  if (!Number.isInteger(index) || index < 1 || index > 129) {
    throw new Error('Invalid frozen case index.')
  }

  const payload = await getVerifiedCaseResponse(
    `frozen-test/cases/${index}`,
    signal,
  )

  if (
    !isRecord(payload) ||
    payload.milestone !== 'M9E6' ||
    typeof payload.evaluation_sha256 !== 'string' ||
    !/^[a-f0-9]{64}$/.test(payload.evaluation_sha256) ||
    !isFrozenCase(payload.case) ||
    payload.case.case_index !== index
  ) {
    throw new Error('Frozen case-detail validation failed.')
  }

  return payload as FrozenCaseDetail
}


export type FrozenInitialPosition = {
  case_index: number
  coordinates: {
    x: number
    y: number
    z: number
  }
  mask_source: 'final' | 'baseline' | 'volume_center'
  coordinate_system: 'native_voxel_xyz'
  selection:
    | 'foreground_nearest_mask_centroid'
    | 'volume_center_fallback'
}

export async function getFrozenInitialPosition(
  caseIndex: number,
  signal?: AbortSignal,
): Promise<FrozenInitialPosition> {
  if (!Number.isInteger(caseIndex) || caseIndex < 1 || caseIndex > 129) {
    throw new Error('Invalid frozen case index.')
  }

  const payload = await getVerifiedCaseResponse(
    `frozen-test/cases/${caseIndex}/initial-position`,
    signal,
  )

  const validAxis = (value: unknown, maximum: number): boolean =>
    typeof value === 'number' &&
    Number.isInteger(value) &&
    value >= 0 &&
    value <= maximum

  if (!isRecord(payload) || !isRecord(payload.coordinates)) {
    throw new Error('Invalid frozen initial-position response.')
  }

  const xyz = payload.coordinates
  const source = payload.mask_source
  const selection = payload.selection

  const validSource =
    source === 'final' ||
    source === 'baseline' ||
    source === 'volume_center'

  const validSelection =
    (source === 'volume_center' &&
      selection === 'volume_center_fallback') ||
    ((source === 'final' || source === 'baseline') &&
      selection === 'foreground_nearest_mask_centroid')

  if (
    payload.case_index !== caseIndex ||
    payload.coordinate_system !== 'native_voxel_xyz' ||
    !validAxis(xyz.x, 181) ||
    !validAxis(xyz.y, 217) ||
    !validAxis(xyz.z, 181) ||
    !validSource ||
    !validSelection
  ) {
    throw new Error('Frozen initial-position validation failed.')
  }

  return payload as FrozenInitialPosition
}

export type FrozenRemovedFocus = {
  case_index: number
  available: boolean
  coordinates: {
    x: number
    y: number
    z: number
  } | null
  removed_voxels: number
  mask_source: 'removed'
  coordinate_system: 'native_voxel_xyz'
  selection:
    | 'foreground_nearest_removed_centroid'
    | 'unavailable_no_removed_voxels'
}

export async function getFrozenRemovedFocus(
  caseIndex: number,
  signal?: AbortSignal,
): Promise<FrozenRemovedFocus> {
  if (
    !Number.isInteger(caseIndex) ||
    caseIndex < 1 ||
    caseIndex > 129
  ) {
    throw new Error('Invalid frozen case index.')
  }

  const payload = await getVerifiedCaseResponse(
    `frozen-test/cases/${caseIndex}/removed-focus`,
    signal,
  )

  if (!isRecord(payload)) {
    throw new Error('Invalid removed-focus response.')
  }

  const count = payload.removed_voxels
  const available = payload.available
  const xyz = payload.coordinates

  const validCount =
    typeof count === 'number' &&
    Number.isInteger(count) &&
    count >= 0 &&
    count <= 182 * 218 * 182

  const validAxis = (value: unknown, maximum: number) =>
    typeof value === 'number' &&
    Number.isInteger(value) &&
    value >= 0 &&
    value <= maximum

  const validAvailable =
    available === true &&
    validCount &&
    count > 0 &&
    isRecord(xyz) &&
    validAxis(xyz.x, 181) &&
    validAxis(xyz.y, 217) &&
    validAxis(xyz.z, 181) &&
    payload.selection ===
      'foreground_nearest_removed_centroid'

  const validUnavailable =
    available === false &&
    count === 0 &&
    xyz === null &&
    payload.selection ===
      'unavailable_no_removed_voxels'

  if (
    payload.case_index !== caseIndex ||
    payload.mask_source !== 'removed' ||
    payload.coordinate_system !== 'native_voxel_xyz' ||
    (!validAvailable && !validUnavailable)
  ) {
    throw new Error('Frozen removed-focus validation failed.')
  }

  return payload as FrozenRemovedFocus
}

export type FrozenMriModality =
  't1n' | 't1c' | 't2w' | 't2f'

export type FrozenAxialLayer =
  'none' | 'baseline' | 'final' | 'removed' | 'support'

export function getFrozenAxialSliceUrl(
  caseIndex: number,
  z: number,
  modality: FrozenMriModality,
  layer: FrozenAxialLayer,
): string {
  if (
    !Number.isInteger(caseIndex) ||
    caseIndex < 1 ||
    caseIndex > 129 ||
    !Number.isInteger(z) ||
    z < 0 ||
    z > 181
  ) {
    throw new Error('Invalid frozen MRI slice coordinates.')
  }

  const params = new URLSearchParams({
    z: String(z),
    modality,
    layer,
  })

  return `${API_BASE}/frozen-test/cases/${caseIndex}/slices/axial?${params}`
}


export type FrozenMriPlane =
  'axial' | 'coronal' | 'sagittal'

export type FrozenContrastMode =
  'slice' | 'volume'

const FROZEN_PLANE_MAX_INDEX: Record<FrozenMriPlane, number> = {
  axial: 181,
  coronal: 217,
  sagittal: 181,
}

export function getFrozenPlaneSliceUrl(
  caseIndex: number,
  plane: FrozenMriPlane,
  index: number,
  modality: FrozenMriModality,
  layer: FrozenAxialLayer,
  contrast: FrozenContrastMode = 'volume',
): string {
  if (
    !Number.isInteger(caseIndex) ||
    caseIndex < 1 ||
    caseIndex > 129 ||
    !Object.prototype.hasOwnProperty.call(
      FROZEN_PLANE_MAX_INDEX, plane,
    ) ||
    !Number.isInteger(index) ||
    index < 0 ||
    index > FROZEN_PLANE_MAX_INDEX[plane]
  ) {
    throw new Error('Invalid frozen MRI plane coordinates.')
  }

  const params = new URLSearchParams({
    modality,
    layer,
    contrast,
    [plane === 'axial' ? 'z' : 'index']: String(index),
  })

  return (
    `${API_BASE}/frozen-test/cases/${caseIndex}` +
    `/slices/${plane}?${params.toString()}`
  )
}
