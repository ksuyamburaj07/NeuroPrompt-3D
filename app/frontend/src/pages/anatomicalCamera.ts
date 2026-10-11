import { Matrix4, Quaternion, Vector3 } from 'three'

import type { MRIViewerMetadata } from '../api/viewer'

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

export const ANATOMICAL_LABELS: Record<string, string> = {
  L: 'Left', R: 'Right', A: 'Anterior', P: 'Posterior', S: 'Superior', I: 'Inferior',
}

export function mriCameraFrame(metadata: MRIViewerMetadata) {
  const centreVoxel = metadata.shape_ras_xyz.map((size) => (size - 1) / 2)
  const world = (voxel: number[]) => new Vector3(...[0, 1, 2].map((row) => (
    metadata.affine_ras[row][0] * voxel[0] +
    metadata.affine_ras[row][1] * voxel[1] +
    metadata.affine_ras[row][2] * voxel[2] +
    metadata.affine_ras[row][3]
  )) as [number, number, number])
  const centre = world(centreVoxel)
  const corners: Vector3[] = []

  // Physical voxel edges also contain the MRI slice planes. Use the entire
  // affine (including obliquity), not spacing or array indices as world axes.
  for (const x of [-0.5, metadata.shape_ras_xyz[0] - 0.5]) {
    for (const y of [-0.5, metadata.shape_ras_xyz[1] - 0.5]) {
      for (const z of [-0.5, metadata.shape_ras_xyz[2] - 0.5]) {
        corners.push(world([x, y, z]).sub(centre))
      }
    }
  }

  return {
    centre: centre.toArray(),
    corners,
    radius: Math.max(1, ...corners.map((corner) => corner.length())),
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

  return { quaternion, distance, up }
}
