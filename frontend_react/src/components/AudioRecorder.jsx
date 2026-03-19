import { useState, useRef, useEffect } from 'react'

export default function AudioRecorder({ onRecorded, disabled }) {
  const [state, setState]     = useState('idle')   // idle | recording | done
  const [duration, setDur]    = useState(0)
  const [audioUrl, setUrl]    = useState(null)
  const [blob, setBlob]       = useState(null)
  const mediaRef = useRef(null)
  const chunksRef= useRef([])
  const timerRef = useRef(null)

  useEffect(() => () => {
    clearInterval(timerRef.current)
    if (mediaRef.current) mediaRef.current.stop()
  }, [])

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      const mr     = new MediaRecorder(stream, { mimeType: 'audio/webm' })
      mediaRef.current = mr
      chunksRef.current = []

      mr.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data) }
      mr.onstop = () => {
        const b = new Blob(chunksRef.current, { type: 'audio/webm' })
        setBlob(b)
        setUrl(URL.createObjectURL(b))
        setState('done')
        stream.getTracks().forEach(t => t.stop())
        clearInterval(timerRef.current)
      }

      mr.start(100)
      setState('recording')
      setDur(0)
      timerRef.current = setInterval(() => setDur(d => d + 1), 1000)
    } catch {
      alert('Microphone access denied. Please allow microphone access in your browser.')
    }
  }

  function stopRecording() {
    if (mediaRef.current && mediaRef.current.state === 'recording') {
      mediaRef.current.stop()
    }
    clearInterval(timerRef.current)
  }

  function discard() {
    setUrl(null)
    setBlob(null)
    setState('idle')
    setDur(0)
  }

  const fmt = s => `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`

  return (
    <div className="audio-recorder">
      {state === 'idle' && (
        <>
          <button
            className="record-btn"
            onClick={startRecording}
            disabled={disabled}
            title="Start recording"
          >
            ♪
          </button>
          <span className="text-sm muted italic">Click to start recording</span>
        </>
      )}

      {state === 'recording' && (
        <>
          <div className="audio-viz">
            {[1,2,3,4,5].map(i => (
              <div key={i} className="audio-bar" style={{ animationDuration: `${0.6 + i * 0.1}s` }} />
            ))}
          </div>
          <button className="record-btn recording" onClick={stopRecording} title="Stop recording">
            ■
          </button>
          <span className="text-sm bold" style={{ color: 'var(--red)', fontFamily: 'var(--font-mono)' }}>
            {fmt(duration)}
          </span>
        </>
      )}

      {state === 'done' && blob && (
        <>
          <audio controls src={audioUrl} style={{ width: '100%', maxWidth: 360 }} />
          <div className="flex gap-sm">
            <button
              className="btn btn-primary"
              onClick={() => onRecorded(blob)}
              disabled={disabled}
            >
              Classify →
            </button>
            <button className="btn btn-outline btn-sm" onClick={discard}>
              Discard
            </button>
          </div>
          <span className="text-xs muted">{fmt(duration)} recorded</span>
        </>
      )}
    </div>
  )
}
