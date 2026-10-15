import assert from 'node:assert/strict'
import { Matrix4, PerspectiveCamera, Quaternion, Vector3 } from 'three'
import { OrbitControls } from 'three-stdlib'

import {
  ANATOMICAL_VIEWS,
  anatomicalCameraPose,
  fittedCameraDistance,
  mriCameraFrame,
  mriCameraLimits,
  mriPlaneForView,
  type CameraView,
} from '../src/pages/anatomicalCamera.ts'

type Metadata = Parameters<typeof mriCameraFrame>[0]
type Point = [number, number, number]

const fov = 43
const failures: string[] = []
let checked = 0
let intermediateClips = 0

function check(name: string, verify: () => void) {
  try {
    verify()
    checked += 1
  } catch (error) {
    failures.push(`${name}: ${error instanceof Error ? error.message : String(error)}`)
  }
}

function near(actual: number, expected: number, message: string, tolerance = 1e-7) {
  assert.ok(Math.abs(actual - expected) < tolerance, `${message}: ${actual} vs ${expected}`)
}

function vectorNear(actual: Vector3, expected: Vector3, message: string) {
  assert.ok(actual.distanceTo(expected) < 1e-7, `${message}: ${actual.toArray()} vs ${expected.toArray()}`)
}

function metadata(shape: Point, affine: number[][]): Metadata {
  return {
    case_id: 'synthetic-camera-geometry-check',
    orientation_convention: 'RAS+',
    original_axis_codes: ['R', 'A', 'S'],
    shape_ras_xyz: shape,
    affine_ras: affine,
    voxel_spacing_ras_mm: [1, 1, 1],
    display_intensity: 'synthetic',
    planes: {
      axial: { axis: 2, count: shape[2], center_index: 0, width: shape[0], height: shape[1] },
      coronal: { axis: 1, count: shape[1], center_index: 0, width: shape[0], height: shape[2] },
      sagittal: { axis: 0, count: shape[0], center_index: 0, width: shape[1], height: shape[2] },
    },
  }
}

const planePairs = [
  ['left', 'sagittal'],
  ['right', 'sagittal'],
  ['anterior', 'coronal'],
  ['posterior', 'coronal'],
  ['superior', 'axial'],
  ['inferior', 'axial'],
  ['home', 'sagittal'],
  ['free', 'sagittal'],
] as const

for (const [view, expected] of planePairs) {
  check(`${view} maps to correct MRI plane`, () => {
    assert.equal(mriPlaneForView(view), expected)
  })
}

const rotation = Math.PI / 7
const fixtures = [
  { name: 'identity', metadata: metadata([240, 240, 155], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) },
  { name: 'translated-anisotropic', metadata: metadata([97, 83, 51], [[2, 0, 0, 82], [0, 1.1, 0, -39], [0, 0, 3.2, 121], [0, 0, 0, 1]]) },
  { name: 'translated-oblique', metadata: metadata([80, 130, 61], [[0.9 * Math.cos(rotation), -1.4 * Math.sin(rotation), 0, -17], [0.9 * Math.sin(rotation), 1.4 * Math.cos(rotation), 0, 48], [0, 0, 2.8, -92], [0, 0, 0, 1]]) },
  { name: 'translated-shear', metadata: metadata([71, 57, 91], [[2, 0.35, 0.2, 200], [0.3, 1.5, -0.15, -100], [0.1, 0.45, 3.2, 27], [0, 0, 0, 1]]) },
  { name: 'reflected-affine', metadata: metadata([43, 82, 61], [[-1.8, 0, 0, 74], [0, 2.3, 0, -46], [0, 0, 1.6, 12], [0, 0, 0, 1]]) },
  { name: 'long-thin-volume', metadata: metadata([3, 5, 1000], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]) },
]

// Expected anatomy is independent of the implementation's preset table.
const expected = {
  left: { direction: [-1, 0, 0], up: [0, 0, 1], edges: ['S', 'I', 'A', 'P'] },
  right: { direction: [1, 0, 0], up: [0, 0, 1], edges: ['S', 'I', 'P', 'A'] },
  anterior: { direction: [0, 1, 0], up: [0, 0, 1], edges: ['S', 'I', 'R', 'L'] },
  posterior: { direction: [0, -1, 0], up: [0, 0, 1], edges: ['S', 'I', 'L', 'R'] },
  superior: { direction: [0, 0, 1], up: [0, 1, 0], edges: ['A', 'P', 'L', 'R'] },
  inferior: { direction: [0, 0, -1], up: [0, 1, 0], edges: ['A', 'P', 'R', 'L'] },
} as const
const anatomicalDirections: Record<string, Point> = {
  R: [1, 0, 0], L: [-1, 0, 0], A: [0, 1, 0], P: [0, -1, 0], S: [0, 0, 1], I: [0, 0, -1],
}
const allViews: CameraView[] = ['home', 'left', 'right', 'anterior', 'posterior', 'superior', 'inferior']
const aspects = [0.08, 0.2, 0.45, 1, 2.4]

function cameraFor(pose: ReturnType<typeof anatomicalCameraPose>, aspect: number) {
  const camera = new PerspectiveCamera(fov, aspect, 0.001, 1_000_000)
  camera.position.set(0, 0, pose.distance).applyQuaternion(pose.quaternion)
  camera.quaternion.copy(pose.quaternion)
  camera.up.copy(pose.up)
  camera.updateMatrixWorld()
  return camera
}

function fits(camera: PerspectiveCamera, corners: Vector3[], margin = 1) {
  camera.updateMatrixWorld()
  for (const [index, corner] of corners.entries()) {
    const projected = corner.clone().project(camera)
    assert.ok(Math.abs(projected.x) <= margin + 1e-7, `corner ${index} horizontal projection ${projected.x}`)
    assert.ok(Math.abs(projected.y) <= margin + 1e-7, `corner ${index} vertical projection ${projected.y}`)
    assert.ok(projected.z >= -1 && projected.z <= 1, `corner ${index} depth projection ${projected.z}`)
  }
}

for (const fixture of fixtures) {
  const frame = mriCameraFrame(fixture.metadata)
  // Row-major affine is independently applied using Three.js matrix math.
  const affine = new Matrix4().fromArray(fixture.metadata.affine_ras.flat()).transpose()
  const shape = fixture.metadata.shape_ras_xyz
  const centre = new Vector3(...shape.map((size) => (size - 1) / 2) as Point).applyMatrix4(affine)
  const independentCorners: Vector3[] = []
  for (const x of [-0.5, shape[0] - 0.5]) {
    for (const y of [-0.5, shape[1] - 0.5]) {
      for (const z of [-0.5, shape[2] - 0.5]) {
        independentCorners.push(new Vector3(x, y, z).applyMatrix4(affine).sub(centre))
      }
    }
  }
  check(`${fixture.name} affine bounds`, () => {
    vectorNear(new Vector3(...frame.centre), centre, 'physical volume centre')
    assert.equal(frame.corners.length, 8)
    frame.corners.forEach((corner, index) => vectorNear(corner, independentCorners[index], `corner ${index}`))
    near(frame.radius, Math.max(...independentCorners.map((corner) => corner.length())), 'volume radius')
  })

  for (const aspect of aspects) {
    const { minDistance, maxDistance, near: nearClip, far: farClip } = mriCameraLimits(frame, fov, aspect)
    for (const view of allViews) {
      const pose = anatomicalCameraPose(view, frame, fov, aspect)
      const camera = cameraFor(pose, aspect)
      camera.near = nearClip
      camera.far = farClip
      camera.updateProjectionMatrix()
      check(`${fixture.name} ${view} aspect ${aspect} fit`, () => {
        assert.ok(pose.distance >= minDistance && pose.distance <= maxDistance, `pose radius ${pose.distance} outside control limits ${minDistance}/${maxDistance}`)
        fits(camera, independentCorners, 1 / 1.12)
      })
      check(`${fixture.name} ${view} aspect ${aspect} controls`, () => {
        const controls = new OrbitControls(camera)
        controls.enableDamping = false
        controls.minDistance = minDistance
        controls.maxDistance = maxDistance
        controls.target.set(0, 0, 0)
        controls.update()
        assert.ok(camera.quaternion.angleTo(pose.quaternion) < 1e-7, 'controls change the requested orientation')
        vectorNear(camera.up, pose.up, 'controls change requested up')
        near(camera.position.length(), pose.distance, 'controls change fitted radius')
        fits(camera, independentCorners, 1 / 1.12)
        controls.dispose()
      })
    }
  }
}

const baseFrame = mriCameraFrame(fixtures[0].metadata)
for (const [id, anatomy] of Object.entries(expected)) {
  check(`${id} camera side, anatomical up, and edge labels`, () => {
    const preset = ANATOMICAL_VIEWS.find((candidate) => candidate.id === id)
    assert.ok(preset)
    assert.deepEqual(preset.edges, anatomy.edges)
    const pose = anatomicalCameraPose(id as CameraView, baseFrame, fov, 1)
    const camera = cameraFor(pose, 1)
    vectorNear(camera.position.clone().normalize(), new Vector3(...anatomy.direction), 'camera side')
    vectorNear(camera.getWorldDirection(new Vector3()), new Vector3(...anatomy.direction).negate(), 'view direction')
    vectorNear(camera.up, new Vector3(...anatomy.up), 'anatomical up')
    const [top, bottom, left, right] = anatomy.edges.map((label) => new Vector3(...anatomicalDirections[label]).multiplyScalar(20).project(camera))
    assert.ok(top.y > 0 && Math.abs(top.x) < 1e-7, 'top anatomical label')
    assert.ok(bottom.y < 0 && Math.abs(bottom.x) < 1e-7, 'bottom anatomical label')
    assert.ok(left.x < 0 && Math.abs(left.y) < 1e-7, 'left anatomical label')
    assert.ok(right.x > 0 && Math.abs(right.y) < 1e-7, 'right anatomical label')
  })
}

const opposing: [CameraView, CameraView][] = [
  ['left', 'right'], ['right', 'left'], ['anterior', 'posterior'],
  ['posterior', 'anterior'], ['superior', 'inferior'], ['inferior', 'superior'],
]
for (const fixture of fixtures) {
  const frame = mriCameraFrame(fixture.metadata)
  for (const aspect of [0.08, 0.45, 1]) {
    const limits = mriCameraLimits(frame, fov, aspect)
    for (const [fromView, toView] of opposing) {
    check(`${fixture.name} ${fromView} to ${toView} aspect ${aspect} orbit path`, () => {
      const from = anatomicalCameraPose(fromView, frame, fov, aspect)
      const to = anatomicalCameraPose(toView, frame, fov, aspect)
      for (let step = 0; step <= 64; step += 1) {
        const progress = step / 64
        const eased = progress * progress * (3 - 2 * progress)
        const quaternion = new Quaternion().slerpQuaternions(from.quaternion, to.quaternion, eased)
        const interpolatedDistance = from.distance + (to.distance - from.distance) * eased
        const fitDistance = fittedCameraDistance(frame, quaternion, fov, aspect)
        const distance = Math.max(interpolatedDistance, fitDistance)
        const position = new Vector3(0, 0, distance).applyQuaternion(quaternion)
        near(quaternion.length(), 1, 'unit transition quaternion')
        near(position.length(), distance, 'orbit radius')
        assert.ok(distance >= Math.min(from.distance, to.distance) - 1e-7, 'opposing path approaches volume centre')
        if (step === 0) assert.ok(quaternion.angleTo(from.quaternion) < 1e-7)
        if (step === 64) assert.ok(quaternion.angleTo(to.quaternion) < 1e-7)
        if (interpolatedDistance + 1e-7 < fitDistance) intermediateClips += 1
        assert.ok(distance <= limits.maxDistance + 1e-7, `intermediate radius ${distance} exceeds controls/far-bound maximum ${limits.maxDistance}`)
        const camera = cameraFor({
          quaternion,
          distance,
          up: new Vector3(0, 1, 0).applyQuaternion(quaternion),
          target: new Vector3(),
        }, aspect)
        camera.near = limits.near
        camera.far = limits.far
        camera.updateProjectionMatrix()
        fits(camera, frame.corners, 1 / 1.12)
      }
    })
    }
  }
}

check('explicit home restores superior and inferior up through controls', () => {
  const camera = cameraFor(anatomicalCameraPose('home', baseFrame, fov, 1), 1)
  const controls = new OrbitControls(camera)
  controls.enableDamping = false
  for (const pole of ['superior', 'inferior'] as const) {
    const pose = anatomicalCameraPose(pole, baseFrame, fov, 1)
    camera.position.set(0, 0, pose.distance).applyQuaternion(pose.quaternion)
    camera.quaternion.copy(pose.quaternion)
    camera.up.copy(pose.up)
    controls.update()
    const home = anatomicalCameraPose('home', baseFrame, fov, 1)
    camera.position.set(0, 0, home.distance).applyQuaternion(home.quaternion)
    camera.quaternion.copy(home.quaternion)
    camera.up.copy(home.up)
    controls.target.set(0, 0, 0)
    controls.update()
    vectorNear(camera.up, new Vector3(0, 0, 1), 'reset RAS up')
    assert.ok(camera.quaternion.angleTo(home.quaternion) < 1e-7, 'home pose normalization')
    near(camera.position.length(), home.distance, 'home fitted radius')
  }
  controls.dispose()
})

check('clearing pending damping prevents drift after a programmatic home pose', () => {
  const home = anatomicalCameraPose('home', baseFrame, fov, 1)
  const camera = cameraFor(home, 1)
  const controls = new OrbitControls(camera)
  controls.enableDamping = true
  controls.dampingFactor = 0.08
  controls.setAzimuthalAngle(controls.getAzimuthalAngle() + 0.7)
  const capturedPosition = camera.position.clone()
  const capturedQuaternion = camera.quaternion.clone()
  const capturedUp = camera.up.clone()
  controls.enableDamping = false
  controls.update()
  controls.enableDamping = true
  camera.position.copy(capturedPosition)
  camera.quaternion.copy(capturedQuaternion)
  camera.up.copy(capturedUp)
  camera.position.set(0, 0, home.distance).applyQuaternion(home.quaternion)
  camera.quaternion.copy(home.quaternion)
  camera.up.copy(home.up)
  controls.target.set(0, 0, 0)
  controls.update()
  for (let frame = 0; frame < 50; frame += 1) controls.update()
  near(camera.position.length(), home.distance, 'stable reset radius')
  assert.ok(camera.quaternion.angleTo(home.quaternion) < 1e-7, 'pending inertia causes post-reset orbit drift')
  vectorNear(camera.up, home.up, 'stable reset up')
  controls.dispose()
})

// Foreground-target framing must not change physical RAS orientation.
for (const fixture of fixtures) {
  const source = fixture.metadata
  const shape = source.shape_ras_xyz
  const lower = shape.map((n) => Math.floor(n * 0.11))
  const upper = shape.map((n, axis) =>
    Math.max(lower[axis] + 1, Math.ceil(n * 0.82))
  )

  const focusedMetadata: Metadata = {
    ...source,
    foreground_bounds_ras_xyz: {
      lower,
      upper_exclusive: upper,
    },
  }

  const full = mriCameraFrame(focusedMetadata, 'volume')
  const focus = mriCameraFrame(focusedMetadata, 'anatomy')

  check(`${fixture.name} foreground frame`, () => {
    assert.equal(full.sourceSignature, focus.sourceSignature)
    vectorNear(
      new Vector3(...full.centre),
      new Vector3(...focus.centre),
      'unchanged MRI scene origin',
    )
    assert.ok(focus.radius < full.radius)
    assert.equal(focus.corners.length, 8)
  })

  for (const aspect of [0.08, 0.45, 1, 2.4]) {
    for (const view of allViews) {
      check(`${fixture.name} ${view} focused aspect ${aspect}`, () => {
        const pose = anatomicalCameraPose(
          view, focus, fov, aspect,
        )
        const camera = cameraFor(pose, aspect)

        // Existing geometry is translated by volume centre.
        // The new camera orbits around the relative foreground centre.
        camera.position.add(pose.target)

        const limits = mriCameraLimits(focus, fov, aspect)
        camera.near = limits.near
        camera.far = limits.far
        camera.updateProjectionMatrix()

        const originalSceneCorners = focus.corners.map(
          (corner) => corner.clone().add(pose.target)
        )

        fits(camera, originalSceneCorners, 1 / 1.12)
      })
    }
  }
}

console.log(`Camera geometry: ${checked} checks passed; ${failures.length} failed.`)
console.log(`Transition framing: ${intermediateClips} sampled poses verified with the extra distance required beyond endpoint interpolation.`)
if (failures.length) {
  console.error(failures.slice(0, 20).join('\n'))
  if (failures.length > 20) console.error(`...and ${failures.length - 20} more failures.`)
  process.exitCode = 1
}
