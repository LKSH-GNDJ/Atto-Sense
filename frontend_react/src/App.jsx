import { useState } from 'react'
import Layout from './components/Layout'
import Classify from './pages/Classify'
import Inbox from './pages/Inbox'
import Analytics from './pages/Analytics'

function genId() { return Math.random().toString(36).slice(2, 10).toUpperCase() }

export default function App() {
  const [page,       setPage]       = useState('classify')
  const [sessionId,  setSessionId]  = useState(genId)
  const [inboxCount, setInboxCount] = useState(0)

  return (
    <Layout page={page} setPage={setPage} inboxCount={inboxCount}
            sessionId={sessionId} onNewSession={() => { setSessionId(genId()); setPage('classify') }}>
      {page === 'classify'  && <Classify  sessionId={sessionId} />}
      {page === 'inbox'     && <Inbox     onInboxCount={setInboxCount} />}
      {page === 'analytics' && <Analytics />}
    </Layout>
  )
}
