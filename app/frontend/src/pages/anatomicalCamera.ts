import { Matrix4, Quaternion, Vector3 } from 'three'

import type { MRIPlane, MRIViewerMetadata } from '../api/viewer'

export const ANATOMICAL_VIEWS = [
  { id: 'left', label: 'Left', code: 'L', direction: [-1, 0, 0], up: [0, 0, 1], edges: ['S', 'I', 'A', 'P'] },
  { id: 'right', label: 'Right', code: 'R', direction: [1, 0, 0], up: [0, 0, 1], edges: ['S', 'I', 'P', 'A'] },
  { id: 'anterior', label: 'Anterior', code: 'A', direction: [0, 1, 0], up: [0, 0, 1], edges: ['S', 'I', 'R', 'L'] },
  { id: 'posterior', label: 'Posterior', code: 'P', direction: [0, -1, 0], up: [0, 0, 1], edges: ['S', 'I', 'L', 'R'] },
  { id: 'superior', label: 'Superior', code: 'S', direction: [0, 0, 1], up: [0, 1, 0], edges: ['A', 'P', 'L', 'R'] },
  { id: 'inferior', label: 'Inferior', code: 'I', direction: [0, 0, -1], up: [0, 1, 0], edges: ['A', 'P', 'R', 'L'] },
] as const

export type AnatomicalView = typeof ANATOMICAL_VIEWS[number]['id']
export type CameraView = AnatomicalView | 'home'
export type CameraViewState = {
  view: CameraView | 'free'
  moving: boolean
}

// Anatomically aligned MRI slice for each camera direction.
// Free orbit and home default to sagittal when inspection begins.
export function mriPlaneForView(
  view: CameraView | 'free',
): MRIPlane {
  if (view === 'anterior' || view === 'posterior') {
    return 'coronal'
  }
  if (view === 'superior' || view === 'inferior') {
    return 'axial'
  }
  return 'sagittal'
}

export const ANATOMICAL_LABELS: Record<string, string> = {
  L: 'Left', R: 'Right', A: 'Anterior', P: 'Posterior', S: 'Superior', I: 'Inferior',
}

export type CameraFitMode = 'volume' | 'anatomy'

export function mriCameraFrame(
  metadata: MRIViewerMetadata,
  fit: CameraFitMode = 'volume',
) {
  const shape = metadata.shape_ras_xyz
  const bounds = fit === 'anatomy'
    ? metadata.foreground_bounds_ras_xyz
    : undefined

  if (fit === 'anatomy' && !bounds) {
    throw new Error('Anatomical foreground bounds are unavailable.')
  }

  const lower = bounds?.lower ?? [0, 0, 0]
  const upper = bounds?.upper_exclusive ?? shape

  if (
    shape.length !== 3 ||
    lower.length !== 3 ||
    upper.length !== 3 ||
    ![0, 1, 2].every((axis) =>
      Number.isInteger(shape[axis]) &&
      shape[axis] > 0 &&
      Number.isInteger(lower[axis]) &&
      Number.isInteger(upper[axis]) &&
      lower[axis] >= 0 &&
      upper[axis] > lower[axis] &&
      upper[axis] <= shape[axis]
    )
  ) {
    throw new Error('Invalid MRI volume or foreground bounds.')
  }

  const world = (voxel: number[]) => new Vector3(
    ...[0, 1, 2].map((row) => (
      metadata.affine_ras[row][0] * voxel[0] +
      metadata.affine_ras[row][1] * voxel[1] +
      metadata.affine_ras[row][2] * voxel[2] +
      metadata.affine_ras[row][3]
    )) as [number, number, number]
  )

  const volumeCentre = world(
    shape.map((size) => (size - 1) / 2)
  )

  const focusCentre = world(
    lower.map((value, axis) =>
      (value + upper[axis] - 1) / 2
    )
  )

  const target = focusCentre.clone().sub(volumeCentre)

  const corners: Vector3[] = []

  // Bounds are voxel-edge coordinates, transformed by the
  // complete MRI affine into physical RAS+ millimetres.
  for (const x of [lower[0] - 0.5, upper[0] - 0.5]) {
    for (const y of [lower[1] - 0.5, upper[1] - 0.5]) {
      for (const z of [lower[2] - 0.5, upper[2] - 0.5]) {
        corners.push(
          world([x, y, z]).sub(focusCentre)
        )
      }
    }
  }

  return {
    centre: volumeCentre.toArray(),
    target: target.toArray(),
    corners,
    radius: Math.max(
      1,
      ...corners.map((corner) => corner.length())
    ),
    sourceSignature: JSON.stringify([
      metadata.case_id,
      metadata.shape_ras_xyz,
      metadata.affine_ras,
    ]),
  }
}

export type MRICameraFrame = ReturnType<typeof mriCameraFrame>

export function mriCameraLimits(frame: MRICameraFrame, fov: number, aspect: number) {
  const verticalTangent = Math.tan(fov * Math.PI / 360)
  const minTangent = verticalTangent * Math.min(1, Math.max(aspect, 0.000001))
  // Bound every orientation, including intermediate roll during pole-to-pole
  // transitions. A home-pose-only bound can clip long, narrow volumes.
  const universalFit = frame.radius * (1 + 1.12 / minTangent)
  const maxDistance = Math.max(frame.radius * 6, universalFit * 3)
  return {
    minDistance: frame.radius * 0.3,
    maxDistance,
    near: Math.max(0.1, frame.radius / 1000),
    far: maxDistance + frame.radius * 2,
  }
}

export function fittedCameraDistance(
  frame: MRICameraFrame,
  quaternion: Quaternion,
  fov: number,
  aspect: number,
) {
  const inverse = quaternion.clone().invert()
  const verticalTangent = Math.tan(fov * Math.PI / 360)
  const horizontalTangent = verticalTangent * Math.max(aspect, 0.000001)
  let distance = frame.radius * 0.2

  for (const corner of frame.corners) {
    const local = corner.clone().applyQuaternion(inverse)
    distance = Math.max(
      distance,
      local.z + 1.12 * Math.abs(local.x) / horizontalTangent,
      local.z + 1.12 * Math.abs(local.y) / verticalTangent,
      local.z + frame.radius * 0.05,
    )
  }

  return distance
}

export function anatomicalCameraPose(
  view: CameraView,
  frame: MRICameraFrame,
  fov: number,
  aspect: number,
) {
  const preset = ANATOMICAL_VIEWS.find((candidate) => candidate.id === view)
  const direction = preset
    ? new Vector3(...preset.direction)
    : new Vector3(128, -157, 125).normalize()
  const up = preset ? new Vector3(...preset.up) : new Vector3(0, 0, 1)
  const quaternion = new Quaternion().setFromRotationMatrix(
    new Matrix4().lookAt(direction, new Vector3(), up),
  )
  const distance = fittedCameraDistance(frame, quaternion, fov, aspect)

  return {
    quaternion,
    distance,
    up,
    target: new Vector3(...frame.target),
  }
}
