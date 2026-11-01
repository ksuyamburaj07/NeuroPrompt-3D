/**
 * Native LAS voxel-to-displayed-MRI pixel projection.
 * Coordinates are native XYZ; image pixel indices are zero-based.
 * Display-only geometry: no inference or ground truth.
 */

export type FrozenCrosshairPlane =
  | 'axial'
  | 'coronal'
  | 'sagittal'

export type NativeVoxelXYZ = Readonly<{
  x: number
  y: number
  z: number
}>

export type CrosshairPixel = Readonly<{
  column: number
  row: number
}>

export function projectCrosshairPixel(
  plane: FrozenCrosshairPlane,
  coordinates: NativeVoxelXYZ,
): CrosshairPixel {
  const { x, y, z } = coordinates

  if (
    !Number.isInteger(x) ||
    !Number.isInteger(y) ||
    !Number.isInteger(z) ||
    x < 0 || x > 181 ||
    y < 0 || y > 217 ||
    z < 0 || z > 181
  ) {
    throw new RangeError('Invalid native XYZ coordinates.')
  }

  if (plane === 'axial') {
    return { column: x, row: 217 - y }
  }

  if (plane === 'coronal') {
    return { column: x, row: 181 - z }
  }

  if (plane === 'sagittal') {
    return { column: y, row: 181 - z }
  }

  throw new RangeError('Unsupported anatomical plane.')
}
