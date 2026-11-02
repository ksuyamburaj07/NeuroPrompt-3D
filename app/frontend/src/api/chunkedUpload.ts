import type { CaseFiles, CaseValidationResponse } from './cases'

const API_BASE = (
  import.meta.env.VITE_API_BASE_URL ||
  (import.meta.env.DEV
    ? 'http://127.0.0.1:8000/api/v1'
    : '/api/v1')
).replace(/\/+$/, '')

const CHUNK_SIZE = 768 * 1024
const STORAGE_KEY = 'np3d.upload.session.v1'

type FileSpec = {
  filename: string
  size: number
  sha256: string
}

type SessionState = {
  session_id: string
  status: string
  offsets: Record<string, number>
  validation?: CaseValidationResponse | null
  token?: string
}

type SavedSession = {
  session_id: string
  token: string
  signature: string
}

async function api<T>(
  path: string,
  options: RequestInit,
): Promise<T> {
  const response = await fetch(
    `${API_BASE}/${path}`,
    { cache: 'no-store', ...options },
  )

  const payload: unknown = await response.json()

  if (!response.ok) {
    const detail = (
      typeof payload === 'object' && payload !== null &&
      'detail' in payload
        ? payload.detail
        : null
    )

    throw new Error(
      typeof detail === 'string'
        ? detail
        : `MRI transfer failed (HTTP ${response.status}).`,
    )
  }

  return payload as T
}

function tokenHeaders(token: string): Record<string, string> {
  return { 'X-NP3D-Session-Token': token }
}

async function fileSha256(file: File): Promise<string> {
  const digest = await crypto.subtle.digest(
    'SHA-256',
    await file.arrayBuffer(),
  )

  return Array.from(new Uint8Array(digest))
    .map(byte => byte.toString(16).padStart(2, '0'))
    .join('')
}

async function transfer(
  session: SavedSession,
  field: string,
  file: File,
  startingOffset: number,
): Promise<void> {
  let offset = startingOffset

  if (
    !Number.isInteger(offset) ||
    offset < 0 ||
    offset > file.size
  ) {
    throw new Error('Invalid persisted MRI upload offset.')
  }

  while (offset < file.size) {
    const chunk = file.slice(offset, offset + CHUNK_SIZE)
    const nextOffset = offset + chunk.size

    let confirmed = false

    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        const result = await api<{ offset: number }>(
          `upload-sessions/${session.session_id}` +
          `/files/${field}?offset=${offset}`,
          {
            method: 'PUT',
            headers: {
              ...tokenHeaders(session.token),
              'Content-Type': 'application/octet-stream',
            },
            body: chunk,
          },
        )

        if (result.offset !== nextOffset) {
          throw new Error('Unexpected server upload offset.')
        }

        offset = nextOffset
        confirmed = true
        break
      } catch (error) {
        const state = await api<SessionState>(
          `upload-sessions/${session.session_id}`,
          {
            method: 'GET',
            headers: tokenHeaders(session.token),
          },
        )

        if (state.offsets[field] === nextOffset) {
          offset = nextOffset
          confirmed = true
          break
        }

        if (
          state.offsets[field] !== offset ||
          attempt === 2
        ) {
          throw error
        }
      }
    }

    if (!confirmed) {
      throw new Error('MRI chunk was not confirmed.')
    }
  }
}

export async function chunkedValidateCase(
  files: CaseFiles,
): Promise<CaseValidationResponse> {
  const selected: Record<string, File> = {
    t1n: files.t1n,
    t1c: files.t1c,
    t2w: files.t2w,
    t2f: files.t2f,
  }

  if (files.segmentation) {
    selected.segmentation = files.segmentation
  }

  const manifest: Record<string, FileSpec> = {}
  let totalBytes = 0

  for (const [field, file] of Object.entries(selected)) {
    if (
      file.size < 1 ||
      file.size > 80 * 1024 * 1024 ||
      !/\.nii(\.gz)?$/i.test(file.name)
    ) {
      throw new Error(`${field}: unsupported NIfTI file or size.`)
    }

    totalBytes += file.size

    manifest[field] = {
      filename: file.name,
      size: file.size,
      sha256: await fileSha256(file),
    }
  }

  if (totalBytes > 400 * 1024 * 1024) {
    throw new Error('Combined MRI case exceeds 400 MiB.')
  }

  const signature = JSON.stringify(manifest)
  let session: SavedSession | null = null
  let state: SessionState | null = null

  const stored = sessionStorage.getItem(STORAGE_KEY)

  if (stored) {
    try {
      const candidate = JSON.parse(stored) as SavedSession

      if (
        candidate.signature === signature &&
        /^up_[0-9a-f]{32}$/.test(candidate.session_id) &&
        typeof candidate.token === 'string'
      ) {
        session = candidate
        state = await api<SessionState>(
          `upload-sessions/${session.session_id}`,
          {
            method: 'GET',
            headers: tokenHeaders(session.token),
          },
        )
      }
    } catch {
      session = null
      state = null
    }
  }

  if (!session) {
    const created = await api<SessionState>(
      'upload-sessions',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ files: manifest }),
      },
    )

    if (!created.token) {
      throw new Error('Backend did not return an upload token.')
    }

    session = {
      session_id: created.session_id,
      token: created.token,
      signature,
    }

    state = created

    sessionStorage.setItem(
      STORAGE_KEY,
      JSON.stringify(session),
    )
  }

  if (!state) {
    throw new Error('Upload session unavailable.')
  }

  if (state.status === 'uploading') {
    for (const [field, file] of Object.entries(selected)) {
      await transfer(
        session,
        field,
        file,
        state.offsets[field] ?? 0,
      )
    }

    state.validation = await api<CaseValidationResponse>(
      `upload-sessions/${session.session_id}/finalize`,
      {
        method: 'POST',
        headers: tokenHeaders(session.token),
      },
    )
  } else if (state.status !== 'finalized') {
    throw new Error('Upload session cannot be resumed.')
  }

  const result = state.validation

  if (
    !result ||
    typeof result.valid !== 'boolean' ||
    typeof result.ready_for_inference !== 'boolean' ||
    !Array.isArray(result.errors)
  ) {
    throw new Error('Invalid MRI validation response.')
  }

  if (result.case_id) {
    sessionStorage.setItem(
      `np3d.case.access.${result.case_id}`,
      JSON.stringify({
        uploadId: session.session_id,
        token: session.token,
      }),
    )
  }

  sessionStorage.removeItem(STORAGE_KEY)
  return result
}
