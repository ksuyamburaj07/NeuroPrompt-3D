import {
  useCallback,
  useEffect,
  useImperativeHandle,
  useLayoutEffect,
  useRef,
  type ComponentRef,
  type Ref,
} from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import { PerspectiveCamera, Quaternion, Vector3 } from 'three'

import {
  anatomicalCameraPose,
  fittedCameraDistance,
  mriCameraLimits,
  type CameraView,
  type CameraViewState,
  type MRICameraFrame,
} from './anatomicalCamera'

export type AnatomicalCameraHandle = {
  navigate: (view: CameraView) => void
}

type Props = {
  ref: Ref<AnatomicalCameraHandle>
  frame: MRICameraFrame
  onViewChange: (state: CameraViewState) => void
}

type Transition = {
  view: CameraView
  from: Quaternion
  fromDistance: number
  fromFitRatio: number
  to: ReturnType<typeof anatomicalCameraPose>
  startedAt: number | null
}

export function AnatomicalCameraControls({ ref, frame, onViewChange }: Props) {
  const get = useThree((state) => state.get)
  const invalidate = useThree((state) => state.invalidate)
  const canvas = useThree((state) => state.gl.domElement)
  const size = useThree((state) => state.size)
  const controlsRef = useRef<ComponentRef<typeof OrbitControls>>(null)
  const transition = useRef<Transition | null>(null)
  const currentView = useRef<CameraView | 'free'>('home')
  const initializedFrame = useRef<MRICameraFrame | null>(null)
  const scratch = useRef({ quaternion: new Quaternion(), offset: new Vector3() })

  const applyPose = useCallback((pose: ReturnType<typeof anatomicalCameraPose>) => {
    const camera = get().camera as PerspectiveCamera
    camera.position.set(0, 0, pose.distance).applyQuaternion(pose.quaternion)
    camera.quaternion.copy(pose.quaternion)
    camera.up.copy(pose.up)
    camera.zoom = 1
    camera.updateProjectionMatrix()
    const controls = controlsRef.current
    if (controls) {
      controls.target.set(0, 0, 0)
      controls.enabled = true
      controls.update()
    }
    invalidate()
  }, [get, invalidate])

  // Clear pending orbit/zoom inertia before taking camera ownership. Updating
  // once without damping drains the controls' private deltas; preserve the pose.
  const clearInertia = useCallback(() => {
    const camera = get().camera as PerspectiveCamera
    const controls = controlsRef.current
    if (!controls) return
    const position = camera.position.clone()
    const quaternion = camera.quaternion.clone()
    const up = camera.up.clone()
    controls.enableDamping = false
    controls.update()
    controls.enableDamping = true
    camera.position.copy(position)
    camera.quaternion.copy(quaternion)
    camera.up.copy(up)
  }, [get])

  const navigate = useCallback((view: CameraView) => {
    const camera = get().camera as PerspectiveCamera
    const controls = controlsRef.current
    if (!controls) return
    clearInertia()
    const to = anatomicalCameraPose(view, frame, camera.fov, camera.aspect)
    currentView.current = view
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    if (reducedMotion) {
      transition.current = null
      applyPose(to)
      onViewChange({ view, moving: false })
      return
    }

    // Orbit around the target, including opposing views. Interpolating XYZ
    // positions instead would send the camera through the volume's centre.
    transition.current = {
      view,
      from: camera.quaternion.clone(),
      fromDistance: camera.position.length(),
      fromFitRatio: Math.min(1, camera.position.length() / fittedCameraDistance(
        frame, camera.quaternion, camera.fov, camera.aspect,
      )),
      to,
      startedAt: null,
    }
    controls.enabled = false
    onViewChange({ view, moving: true })
    invalidate()
  }, [applyPose, get, clearInertia, frame, invalidate, onViewChange])

  useImperativeHandle(ref, () => ({ navigate }), [navigate])

  useLayoutEffect(() => {
    const camera = get().camera as PerspectiveCamera
    const controls = controlsRef.current
    if (!controls) return
    const home = anatomicalCameraPose('home', frame, camera.fov, camera.aspect)
    const limits = mriCameraLimits(frame, camera.fov, camera.aspect)
    controls.minDistance = limits.minDistance
    controls.maxDistance = limits.maxDistance
    camera.near = limits.near
    camera.far = limits.far
    camera.updateProjectionMatrix()

    if (initializedFrame.current !== frame) {
      initializedFrame.current = frame
      transition.current = null
      currentView.current = 'home'
      clearInertia()
      applyPose(home)
      onViewChange({ view: 'home', moving: false })
    } else if (transition.current) {
      transition.current.to = anatomicalCameraPose(
        transition.current.view, frame, camera.fov, camera.aspect,
      )
    } else if (currentView.current !== 'free') {
      // A named, fitted view stays fitted when its canvas changes size.
      applyPose(anatomicalCameraPose(currentView.current, frame, camera.fov, camera.aspect))
    }
    invalidate()
  }, [applyPose, clearInertia, get, frame, invalidate, onViewChange, size.width, size.height])

  const freeOrbit = useCallback(() => {
    transition.current = null
    currentView.current = 'free'
    const controls = controlsRef.current
    if (controls) controls.enabled = true
    onViewChange({ view: 'free', moving: false })
    invalidate()
  }, [invalidate, onViewChange])

  useEffect(() => {
    // Capture before OrbitControls checks enabled, so input can interrupt an
    // animation and the same pointer/wheel event still reaches the controls.
    canvas.addEventListener('pointerdown', freeOrbit, true)
    canvas.addEventListener('wheel', freeOrbit, { capture: true, passive: true })
    return () => {
      canvas.removeEventListener('pointerdown', freeOrbit, true)
      canvas.removeEventListener('wheel', freeOrbit, true)
    }
  }, [canvas, freeOrbit])

  useFrame((state) => {
    const camera = state.camera as PerspectiveCamera
    const active = transition.current
    if (!active) return
    const now = performance.now()
    // Begin on the first requested frame, even after a long idle period.
    active.startedAt ??= now
    const progress = Math.min(1, (now - active.startedAt) / 450)
    const eased = progress * progress * (3 - 2 * progress)
    const { quaternion, offset } = scratch.current
    quaternion.slerpQuaternions(active.from, active.to.quaternion, eased)
    const interpolatedDistance = active.fromDistance + (active.to.distance - active.fromDistance) * eased
    // Keep fitted views visible throughout their arc. If the user was zoomed
    // in, expand the framing gradually rather than jumping on the first frame.
    const fitRatio = active.fromFitRatio + (1 - active.fromFitRatio) * eased
    const distance = Math.max(interpolatedDistance, fitRatio * fittedCameraDistance(
      frame, quaternion, camera.fov, camera.aspect,
    ))
    camera.position.copy(offset.set(0, 0, distance).applyQuaternion(quaternion))
    camera.quaternion.copy(quaternion)
    camera.up.set(0, 1, 0).applyQuaternion(quaternion)

    if (progress === 1) {
      transition.current = null
      applyPose(active.to)
      onViewChange({ view: active.view, moving: false })
    } else {
      invalidate()
    }
  })

  return (
    <OrbitControls
      ref={controlsRef}
      makeDefault
      enableDamping
      dampingFactor={0.08}
      enablePan={false}
      target={[0, 0, 0]}
    />
  )
}
