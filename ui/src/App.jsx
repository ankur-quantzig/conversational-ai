import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  BotMessageSquare,
  BookOpen,
  ChevronLeft,
  ChevronRight,
  Check,
  CircleUserRound,
  Copy,
  Download,
  Menu,
  RotateCcw,
  Send,
  Share2,
  SquarePen,
  ThumbsDown,
  ThumbsUp,
  X,
} from 'lucide-react'
import logoUrl from './assets/shell-logo.png'

const chunkFiles = import.meta.glob('../output/chunks/*-chunks.jsonl', {
  eager: true,
  import: 'default',
  query: '?raw',
})

const starterExamples = [
  'What are AI agents and how do they work?',
  'How do AI agents use tools and memory?',
  'What is the difference between generative AI, AI agents, and agentic AI?',
  'What are the key ideas across the indexed PDFs and videos?',
]

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api'
const UNABLE_TO_GENERATE_MESSAGE = 'I am unable to generate the response at the moment. Please contact Admin.'

function parseChunks() {
  return Object.entries(chunkFiles).flatMap(([path, raw]) => {
    return String(raw)
      .split('\n')
      .filter(Boolean)
      .map((line) => JSON.parse(line))
      .map((chunk) => ({
        ...chunk,
        file_path: path,
        title: titleFromDocId(chunk.doc_id),
      }))
  })
}

function titleFromDocId(docId = '') {
  return docId
    .replace(/^\d{4}-\d{5}v\d+-/, '')
    .replace(/-/g, ' ')
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function tokenize(text) {
  return String(text)
    .toLowerCase()
    .replace(/[^a-z0-9\s-]/g, ' ')
    .split(/\s+/)
    .filter((word) => word.length > 2)
}

function retrieve(chunks, question, topK, activeDoc, sourceType) {
  const queryTerms = tokenize(question)
  const uniqueTerms = [...new Set(queryTerms)]

  return chunks
    .filter((chunk) => {
      const chunkSourceType = chunk.source_type ?? chunk.metadata?.source_type ?? 'document'
      return (!sourceType || sourceType === 'all' || chunkSourceType === sourceType) && (activeDoc === 'all' || chunk.doc_id === activeDoc)
    })
    .map((chunk) => {
      const text = `${chunk.section ?? ''} ${chunk.content ?? ''}`.toLowerCase()
      const score = uniqueTerms.reduce((total, term) => {
        const matches = text.split(term).length - 1
        return total + matches * (term.length > 5 ? 1.4 : 1)
      }, 0)
      const roleBoost = chunk.role === 'title' || chunk.content_type === 'heading' ? 0.8 : 0
      return { ...chunk, score: score + roleBoost }
    })
    .filter((chunk) => chunk.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
}

function composeAnswer(question, sources) {
  return UNABLE_TO_GENERATE_MESSAGE
}

function localHeading(question, sources) {
  return ''
}

function formatApiMessage(message) {
  const createdAt = message.created_at ? new Date(message.created_at) : null
  const metadata = message.metadata ?? {}
  return {
    id: message.id,
    role: message.role,
    content: message.content,
    heading: metadata.heading ?? '',
    time: createdAt ? new Intl.DateTimeFormat([], { hour: '2-digit', minute: '2-digit' }).format(createdAt) : '',
    sources: metadata.sources ?? [],
    citations: metadata.citations ?? [],
    followUpQuestions: metadata.follow_up_questions ?? [],
    diagram: metadata.diagram ?? null,
    questionAnalysis: metadata.question_analysis ?? null,
    elapsedMs: metadata.elapsed_ms ?? null,
    status: message.role === 'assistant' ? 'final_response' : undefined,
    progressSteps: [],
  }
}

function formatElapsed(milliseconds) {
  if (!Number.isFinite(milliseconds)) return ''
  if (milliseconds < 1000) return `${Math.max(1, Math.round(milliseconds))} ms`
  const seconds = milliseconds / 1000
  return `${seconds < 10 ? seconds.toFixed(1) : Math.round(seconds)} sec`
}

function parseSseBlock(block) {
  let event = 'message'
  const data = []
  block.split(/\r?\n/).forEach((line) => {
    if (line.startsWith('event:')) {
      event = line.slice(6).trim()
    } else if (line.startsWith('data:')) {
      data.push(line.slice(5).trimStart())
    }
  })
  return { event, data: data.length ? JSON.parse(data.join('\n')) : {} }
}

function apiError(status, message) {
  const error = new Error(message || `API request failed: ${status}`)
  error.status = status
  error.userMessage = message
  return error
}

function protectedFailure(status) {
  return [400, 401, 403, 429].includes(status)
}

function safeAgentErrorMessage(error) {
  const requestId = error?.requestId ? ` Request ID: ${error.requestId}.` : ''
  return `${error?.userMessage || error?.message || 'The API could not complete this request.'}${requestId}`
}

function sourceName(source = {}) {
  const candidate = source.title || source.metadata?.title || source.source_pdf || source.source_path || source.doc_id
  if (!candidate) return 'Source document'
  const filename = String(candidate).split(/[\\/]/).pop()
  return filename
    .replace(/\.[a-z0-9]+$/i, '')
    .replace(/^\d{4}-\d{2}-\d{2}-/, '')
    .replace(/^\d{4}-\d{5}v\d+-/, '')
    .replace(/[-_]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function citationLocations(source, citation, index) {
  const isVideo = source.source_type === 'video'
  const timeLabel =
    source.start_time_label && source.end_time_label
      ? `${source.start_time_label} - ${source.end_time_label}`
      : null
  const pages = source.page_numbers?.length ? source.page_numbers : citation?.pages
  if (isVideo && timeLabel) return [timeLabel]
  if (pages?.length) return pages.map((page) => `p. ${page}`)
  return [`source ${index + 1}`]
}

function numericTime(value) {
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function VideoClipPlayer({ source }) {
  const videoRef = useRef(null)
  if (source.source_type !== 'video' || !source.doc_id) return null

  const start = numericTime(source.start_time ?? source.metadata?.start_time) ?? 0
  const end = numericTime(source.end_time ?? source.metadata?.end_time)
  const startLabel = source.start_time_label ?? source.metadata?.start_time_label ?? formatVideoTime(start)
  const endLabel = source.end_time_label ?? source.metadata?.end_time_label ?? (end !== null ? formatVideoTime(end) : '')
  const clipLabel = endLabel ? `${startLabel} - ${endLabel}` : `From ${startLabel}`
  const encodedDocId = encodeURIComponent(source.doc_id)
  const videoUrl =
    end !== null
      ? `${API_BASE_URL}/media/video-clips/${encodedDocId}?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`
      : `${API_BASE_URL}/media/videos/${encodedDocId}#t=${start}`

  function handleLoadedMetadata(event) {
    if (end === null) {
      event.currentTarget.currentTime = start
    }
  }

  function handleTimeUpdate(event) {
    if (end === null) return
    if (event.currentTarget.currentTime >= end) {
      event.currentTarget.pause()
    }
  }

  return (
    <div className="video-clip">
      <div className="video-clip__meta">
        <span>Relevant clip</span>
        <strong>{clipLabel}</strong>
      </div>
      <video
        ref={videoRef}
        controls
        preload="metadata"
        src={videoUrl}
        onLoadedMetadata={handleLoadedMetadata}
        onTimeUpdate={handleTimeUpdate}
      />
    </div>
  )
}

function formatVideoTime(seconds) {
  const totalSeconds = Math.max(0, Math.floor(Number(seconds) || 0))
  const minutes = Math.floor(totalSeconds / 60)
  const remainder = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}

function sourceDisplayContent(source = {}) {
  const content = String(source.content || '').trim()
  if (source.source_type === 'video') return ''
  return content
}

function sourceTypeLabel(source = {}) {
  return source.source_type === 'video' ? 'Video' : 'PDF/Text'
}

function CitationList({ items, openSource, onToggle }) {
  if (!items.length) return null
  const groups = []
  items.forEach((item, index) => {
    const name = sourceName(item.source)
    const type = sourceTypeLabel(item.source)
    let group = groups.find((entry) => entry.name === name && entry.type === type)
    if (!group) {
      group = { name, type, entries: [] }
      groups.push(group)
    }
    citationLocations(item.source, item.citation, index).forEach((label) => {
      if (!group.entries.some((entry) => entry.label === label)) {
        group.entries.push({ ...item, index, label })
      }
    })
  })

  return (
    <div className="citation-list" aria-label="Citations">
      {groups.map((group) => (
        <div className="citation-group" key={`${group.type}-${group.name}`}>
          <span className="citation-prefix">Sources:</span>
          <span className={`citation-type citation-type--${group.type === 'Video' ? 'video' : 'document'}`}>{group.type}</span>
          <span className="citation-source">{group.name}</span>
          <span className="citation-pages">
            {group.entries.map((entry) => (
              <button
                className="citation-link"
                key={`${entry.source?.id ?? 'source'}-${entry.index}`}
                type="button"
                onClick={() => onToggle(entry.index)}
                aria-expanded={openSource === entry.index}
              >
                {entry.label}
              </button>
            ))}
          </span>
        </div>
      ))}
    </div>
  )
}

function SourceChunk({ source, citation, index, onClose }) {
  const pages = source.page_numbers?.length ? source.page_numbers : citation?.pages
  const pageLabel =
    source.source_type === 'video'
      ? citationLocations(source, citation, index).join(', ')
      : pages?.length
        ? `Page ${pages.join(', ')}`
        : 'Document source'
  const documentName = sourceName(source)
  const displayContent = sourceDisplayContent(source)

  return (
    <article className="source-chunk">
      <div className="source-card__meta">
        <span>Source {index + 1}</span>
        <div className="source-card__actions">
          <strong>{pageLabel}</strong>
          {onClose ? (
            <button type="button" aria-label="Close source" onClick={onClose}>
              <X size={14} />
            </button>
          ) : null}
        </div>
      </div>
      <h3>{documentName}</h3>
      {source.source_type !== 'video' && source.section ? <h4>{source.section}</h4> : null}
      <VideoClipPlayer source={source} />
      {source.source_type !== 'video' && citation?.quote ? <blockquote>{citation.quote}</blockquote> : null}
      {displayContent ? (
        <p>{displayContent}</p>
      ) : null}
    </article>
  )
}

function StructuredContent({ text, children }) {
  const blocks = []
  let bullets = []
  let paragraph = []

  function flushParagraph(key) {
    if (!paragraph.length) return
    blocks.push(<p key={key}>{paragraph.join(' ')}</p>)
    paragraph = []
  }

  function flushBullets(key) {
    if (!bullets.length) return
    blocks.push(
      <ul key={key}>
        {bullets.map((bullet, index) => (
          <li key={`${key}-${index}`}>{bullet}</li>
        ))}
      </ul>,
    )
    bullets = []
  }

  String(text || '')
    .split('\n')
    .forEach((line, index) => {
      const trimmed = line.trim()
      if (!trimmed) {
        flushParagraph(`p-${index}`)
        flushBullets(`b-${index}`)
        return
      }
      const bullet = trimmed.match(/^[-*]\s+(.+)$/) || trimmed.match(/^\d+\.\s+(.+)$/)
      if (bullet) {
        flushParagraph(`p-${index}`)
        bullets.push(bullet[1])
      } else {
        flushBullets(`b-${index}`)
        paragraph.push(trimmed)
      }
    })

  flushParagraph('p-last')
  flushBullets('b-last')

  return (
    <div className="structured-content">
      {blocks.length ? blocks : <p>{text}</p>}
      {children}
    </div>
  )
}

function parseMermaidFlow(code = '') {
  const nodes = new Map()
  const edges = []
  const nodePattern = /^([A-Za-z0-9_]+)(?:\[[^\]]*"]?([^"\]]+)["]?\]|\(([^)]+)\)|\{([^}]+)\})?$/

  function cleanLabel(value, fallback) {
    return String(value || fallback || 'Step')
      .replace(/^["']|["']$/g, '')
      .replace(/<br\s*\/?>/gi, ' ')
      .trim()
  }

  function readNode(raw) {
    const normalized = String(raw || '').trim().replace(/;$/, '')
    const match = normalized.match(nodePattern)
    if (!match) return { id: normalized, label: cleanLabel(normalized, normalized) }
    const [, id, bracketLabel, roundLabel, braceLabel] = match
    return { id, label: cleanLabel(bracketLabel || roundLabel || braceLabel, id) }
  }

  String(code)
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line && !line.startsWith('%%') && !line.toLowerCase().startsWith('flowchart') && !line.toLowerCase().startsWith('graph'))
    .forEach((line) => {
      const parts = line.split(/-->|---|==>/)
      if (parts.length < 2) return
      const from = readNode(parts[0])
      const to = readNode(parts[1])
      nodes.set(from.id, from.label)
      nodes.set(to.id, to.label)
      edges.push({ from: from.id, to: to.id })
    })

  return { nodes, edges }
}

function ResponseDiagram({ diagram }) {
  const { nodes, edges } = parseMermaidFlow(diagram.code)
  if (!nodes.size || !edges.length) {
    return (
      <section className="response-diagram">
        {diagram.title ? <h3>{diagram.title}</h3> : null}
        <pre>{diagram.code}</pre>
      </section>
    )
  }

  return (
    <section className="response-diagram" aria-label={diagram.title || 'Response diagram'}>
      {diagram.title ? <h3>{diagram.title}</h3> : null}
      <div className="diagram-flow">
        {edges.map((edge, index) => (
          <React.Fragment key={`${edge.from}-${edge.to}-${index}`}>
            {index === 0 ? <div className="diagram-node">{nodes.get(edge.from) || edge.from}</div> : null}
            <div className="diagram-arrow" aria-hidden="true">↓</div>
            <div className="diagram-node">{nodes.get(edge.to) || edge.to}</div>
          </React.Fragment>
        ))}
      </div>
    </section>
  )
}

function ChatMessage({ message, onCopy, onFeedback, onShare, onDownload, onRetry, isSending }) {
  const isUser = message.role === 'user'
  const Icon = isUser ? CircleUserRound : BotMessageSquare
  const [openSource, setOpenSource] = useState(null)
  const [showSources, setShowSources] = useState(false)
  const citations = !isUser ? buildDisplayCitations(message) : []
  const activeCitation = openSource === null ? null : citations[openSource]
  const stageLabel = {
    agent_ready: 'Agent is ready',
    thinking: 'Thinking',
    checking_access: 'Checking',
    guardrail_check: 'Checking',
    question_analysis: 'Context',
    question_ready: 'Context',
    creating_subquestions: 'Planning',
    subquestions_ready: 'Planning',
    bm25_search: 'Searching',
    vector_search: 'Searching',
    cache_hit: 'Cached',
    deduplicating_chunks: 'Merging',
    reranking_chunks: 'Reranking',
    generating_answer: 'Generating',
    creating_followups: 'Follow-ups',
    diagram_check: 'Diagram',
    saving_conversation: 'Saving',
    complete: 'Complete',
    writing: 'Writing',
    final_response: 'Final response',
    error: 'Needs attention',
  }[message.status]
  const actionsDisabled = message.pending || message.status === 'error'
  const toggleSource = (index) => setOpenSource(openSource === index ? null : index)

  return (
    <article className={`message ${isUser ? 'message--user' : 'message--assistant'}`}>
      <div className="avatar" aria-hidden="true">
        <Icon size={17} />
      </div>
      <div className="message__content">
        <div className="message__top">
          <strong>{isUser ? 'Question' : 'Agent'}</strong>
          {!isUser && stageLabel ? <span className={`stage-pill stage-pill--${message.status}`}>{stageLabel}</span> : null}
          {!isUser && message.elapsedMs ? <span className="response-time">{formatElapsed(message.elapsedMs)}</span> : null}
          {message.time ? <time>{message.time}</time> : null}
        </div>
        {!isUser && message.heading && !message.pending ? <h2 className="response-heading">{message.heading}</h2> : null}
        {!isUser && message.progressSteps?.length && message.pending ? <AgentRunTimeline steps={message.progressSteps} activeStatus={message.status} /> : null}
        {message.content ? (
          <StructuredContent text={message.content}>
            <CitationList items={citations} openSource={openSource} onToggle={toggleSource} />
          </StructuredContent>
        ) : !isUser && message.pending ? (
          <div className="answer-placeholder" aria-hidden="true">
            Waiting for response stream...
          </div>
        ) : null}
        {!isUser && message.diagram?.should_show ? <ResponseDiagram diagram={message.diagram} /> : null}
        {activeCitation ? (
          <SourceChunk
            source={activeCitation.source}
            citation={activeCitation.citation}
            index={openSource}
            onClose={() => setOpenSource(null)}
          />
        ) : null}
        {showSources ? (
          <div className="source-stack">
            <div className="source-stack__top">
              <span>Sources</span>
              <button type="button" aria-label="Close sources" onClick={() => setShowSources(false)}>
                <X size={15} />
              </button>
            </div>
            {citations.map((item, index) => (
              <SourceChunk key={`${item.source?.id ?? 'source'}-all-${index}`} source={item.source} citation={item.citation} index={index} />
            ))}
          </div>
        ) : null}
        {!isUser ? (
          <div className="message-actions" aria-label="Message actions">
            <button type="button" aria-label="Copy response" disabled={actionsDisabled} onClick={() => onCopy(message.content)}>
              <Copy size={15} />
            </button>
            <button type="button" aria-label="Share response" disabled={actionsDisabled || !message.id} onClick={() => onShare(message)}>
              <Share2 size={15} />
            </button>
            <button type="button" aria-label="Download conversation" disabled={message.pending} onClick={() => onDownload()}>
              <Download size={15} />
            </button>
            <button type="button" aria-label="Good response" disabled={actionsDisabled || !message.id} onClick={() => onFeedback(message, 'up')}>
              <ThumbsUp size={15} />
            </button>
            <button type="button" aria-label="Bad response" disabled={actionsDisabled || !message.id} onClick={() => onFeedback(message, 'down')}>
              <ThumbsDown size={15} />
            </button>
            <button type="button" aria-label="Try again" disabled={actionsDisabled || !message.id || isSending} onClick={() => onRetry(message)}>
              <RotateCcw size={15} />
            </button>
            <button type="button" aria-label="Show sources" disabled={actionsDisabled || !citations.length} onClick={() => setShowSources((value) => !value)}>
              <BookOpen size={15} />
            </button>
          </div>
        ) : null}
      </div>
    </article>
  )
}

function AgentRunTimeline({ steps }) {
  const visibleSteps = steps.slice(-8)
  return (
    <div className="agent-run" aria-label="Agent progress">
      {visibleSteps.map((step, index) => {
        const isLast = index === visibleSteps.length - 1
        return (
          <div className={`agent-run__step ${isLast ? 'is-active' : 'is-complete'}`} key={`${step.stage}-${index}`}>
            <span className="agent-run__dot" />
            <span>{step.message}</span>
          </div>
        )
      })}
    </div>
  )
}

function buildDisplayCitations(message) {
  const sources = message.sources ?? []
  const citations = message.citations ?? []
  if (citations.length) {
    return citations
      .map((citation) => {
        const source = sources[citation.source_index - 1]
        return source ? { citation, source } : null
      })
      .filter(Boolean)
  }
  return sources.slice(0, 4).map((source) => ({ citation: null, source }))
}

function SuggestedQuestions({ examples: suggestedExamples, disabled, onSelect, followUp = false }) {
  if (!suggestedExamples.length) return null
  if (followUp) {
    return (
      <div className="prompt-section prompt-section--followup">
        <span>Suggested follow-up questions</span>
        <div className="followup-list">
          {suggestedExamples.slice(0, 4).map((example) => (
            <button className="followup-question" key={example} type="button" disabled={disabled} onClick={() => onSelect(example)}>
              {example}
            </button>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="prompt-section prompt-section--starter">
      <span>Suggested questions</span>
      <div className="prompt-grid">
        {suggestedExamples.map((example) => (
          <button key={example} type="button" disabled={disabled} onClick={() => onSelect(example)}>
            {example}
          </button>
        ))}
      </div>
    </div>
  )
}

export function App() {
  const chunks = useMemo(() => parseChunks(), [])
  const localDocuments = useMemo(() => {
    const byDoc = new Map()
    chunks.forEach((chunk) => {
      const doc = byDoc.get(chunk.doc_id) ?? {
        id: chunk.doc_id,
        title: chunk.title,
        source_type: chunk.source_type ?? chunk.metadata?.source_type ?? 'document',
        chunks: 0,
        pages: new Set(),
      }
      doc.chunks += 1
      ;(chunk.page_numbers ?? []).forEach((page) => doc.pages.add(page))
      byDoc.set(chunk.doc_id, doc)
    })
    return [...byDoc.values()].map((doc) => ({ ...doc, pages: doc.pages.size }))
  }, [chunks])

  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [activeDoc, setActiveDoc] = useState('all')
  const [sessionId, setSessionId] = useState(null)
  const [sessions, setSessions] = useState([])
  const [apiDocuments, setApiDocuments] = useState([])
  const [apiOnline, setApiOnline] = useState(false)
  const [isSending, setIsSending] = useState(false)
  const [copied, setCopied] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const inputRef = useRef(null)

  const documents = apiDocuments.length ? apiDocuments : localDocuments
  const activeDocument = documents.find((doc) => doc.id === activeDoc)
  const latestFollowUps =
    [...messages]
      .reverse()
      .find((message) => message.role === 'assistant' && !message.pending && message.followUpQuestions?.length)
      ?.followUpQuestions ?? []

  useEffect(() => {
    refreshSessions()
  }, [])

  async function apiFetch(path, options) {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      headers: { 'Content-Type': 'application/json', ...(options?.headers ?? {}) },
      ...options,
    })
    if (!response.ok) {
      let message = `API request failed: ${response.status}`
      try {
        const payload = await response.json()
        message = payload.detail || payload.message || message
      } catch {
        // Keep the generic message when the server returned non-JSON HTML/text.
      }
      throw apiError(response.status, message)
    }
    return response.json()
  }

  async function refreshSessions() {
    try {
      const [sessionData, documentData] = await Promise.all([apiFetch('/sessions'), apiFetch('/documents')])
      setSessions(sessionData)
      setApiDocuments(documentData)
      setApiOnline(true)
    } catch {
      setApiOnline(false)
    }
  }

  async function loadSession(nextSessionId) {
    try {
      const data = await apiFetch(`/sessions/${nextSessionId}/messages`)
      setSessionId(nextSessionId)
      setMessages(data.map(formatApiMessage))
      setApiOnline(true)
      setSidebarOpen(false)
    } catch {
      setApiOnline(false)
    }
  }

  function startNewChat() {
    setSessionId(null)
    setMessages([])
    setSidebarOpen(false)
    inputRef.current?.focus()
  }

  function updateMessage(messageId, patch) {
    setMessages((current) => current.map((message) => (message.id === messageId ? { ...message, ...patch } : message)))
  }

  function appendAnswerDelta(messageId, delta) {
    setMessages((current) =>
      current.map((message) => {
        if (message.id !== messageId) return message
        const currentContent = message.streamingStarted ? message.content : ''
        return {
          ...message,
          content: `${currentContent}${delta}`,
          status: 'writing',
          pending: true,
          streamingStarted: true,
        }
      }),
    )
  }

  function appendProgressStep(messageId, step) {
    setMessages((current) =>
      current.map((message) => {
        if (message.id !== messageId) return message
        const existingSteps = message.progressSteps ?? []
        const previous = existingSteps[existingSteps.length - 1]
        const nextStep = {
          stage: step.stage || 'thinking',
          message: step.message || 'Working',
          metadata: step.metadata ?? {},
        }
        if (previous?.stage === nextStep.stage && previous?.message === nextStep.message) {
          return {
            ...message,
            elapsedMs: step.elapsedMs ?? message.elapsedMs,
            status: nextStep.stage,
            pending: true,
          }
        }
        return {
          ...message,
          progressSteps: [...existingSteps, nextStep],
          elapsedMs: step.elapsedMs ?? message.elapsedMs,
          status: nextStep.stage,
          pending: true,
        }
      }),
    )
  }

  function requestPayload(question) {
    return {
      question,
      session_id: sessionId,
      doc_id: activeDoc === 'all' ? null : activeDoc,
      source_type: null,
    }
  }

  async function submitChatJson(payload, assistantId, startedAt, time) {
    const data = await apiFetch('/chat', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
    const elapsedMs = data.elapsed_ms ?? performance.now() - startedAt
    setSessionId(data.session_id)
    setApiOnline(true)
    updateMessage(assistantId, {
      id: data.message_id ?? assistantId,
      role: 'assistant',
      heading: data.heading ?? '',
      content: data.answer,
      time,
      sources: data.sources ?? [],
      citations: data.citations ?? [],
      followUpQuestions: data.follow_up_questions ?? [],
      diagram: data.diagram ?? null,
      questionAnalysis: data.question_analysis ?? null,
      elapsedMs,
      status: 'final_response',
      pending: false,
      progressSteps: [],
    })
    return data
  }

  async function submitChatStream(payload, assistantId, startedAt, time) {
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
    if (!response.ok || !response.body) {
      throw apiError(response.status, `Streaming failed: ${response.status}`)
    }

    setApiOnline(true)
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ''
    let finalPayload = null
    let terminalEvent = false

    function handleBlock(block) {
      if (!block.trim()) return
      const { event, data } = parseSseBlock(block)
      if (event === 'status') {
        updateMessage(assistantId, {
          heading: '',
          status: data.stage || 'thinking',
          pending: true,
          elapsedMs: performance.now() - startedAt,
        })
      }
      if (event === 'progress') {
        appendProgressStep(assistantId, {
          ...data,
          elapsedMs: performance.now() - startedAt,
        })
      }
      if (event === 'answer_delta') {
        appendAnswerDelta(assistantId, data.delta || '')
      }
      if (event === 'final') {
        const elapsedMs = data.elapsed_ms ?? performance.now() - startedAt
        finalPayload = data
        terminalEvent = true
        setSessionId(data.session_id)
        updateMessage(assistantId, {
          id: data.message_id ?? assistantId,
          role: 'assistant',
          heading: data.heading ?? '',
          content: data.answer,
          time,
          sources: data.sources ?? [],
          citations: data.citations ?? [],
          followUpQuestions: data.follow_up_questions ?? [],
          diagram: data.diagram ?? null,
          questionAnalysis: data.question_analysis ?? null,
          elapsedMs,
          status: 'final_response',
          pending: false,
          streamingStarted: false,
          progressSteps: [],
        })
      }
      if (event === 'error') {
        terminalEvent = true
        const error = apiError(data.status_code || 500, data.message || 'The streaming request failed before the agent returned a final response.')
        error.requestId = data.request_id
        error.fromStreamEvent = true
        updateMessage(assistantId, {
          heading: '',
          content: safeAgentErrorMessage(error),
          time,
          sources: [],
          citations: [],
          elapsedMs: data.elapsed_ms ?? performance.now() - startedAt,
          status: 'error',
          pending: false,
          progressSteps: [],
        })
        throw error
      }
    }

    while (true) {
      const { done, value } = await reader.read()
      buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
      const blocks = buffer.split('\n\n')
      buffer = blocks.pop() ?? ''
      blocks.forEach(handleBlock)
      if (done) break
    }
    handleBlock(buffer)

    if (!terminalEvent) {
      throw apiError(500, 'Streaming ended before the agent returned a final response.')
    }
    return finalPayload
  }

  async function submitQuestion(questionText = input) {
    const question = questionText.trim()
    if (!question || isSending) return

    const time = new Intl.DateTimeFormat([], { hour: '2-digit', minute: '2-digit' }).format(new Date())
    const assistantId = crypto.randomUUID()
    const startedAt = performance.now()
    const payload = requestPayload(question)
    setMessages((current) => [
      ...current,
      { id: crypto.randomUUID(), role: 'user', content: question, time, sources: [] },
      {
        id: assistantId,
        role: 'assistant',
        content: '',
        heading: '',
        time,
        sources: [],
        citations: [],
        diagram: null,
        elapsedMs: 0,
        status: 'agent_ready',
        pending: true,
        progressSteps: [{ stage: 'agent_ready', message: 'Agent is ready', metadata: {} }],
      },
    ])
    setInput('')
    setIsSending(true)

    try {
      const streamed = await submitChatStream(payload, assistantId, startedAt, time)
      if (streamed) refreshSessions()
    } catch (streamError) {
      if (streamError.fromStreamEvent) {
        return null
      }
      try {
        const data = await submitChatJson(payload, assistantId, startedAt, time)
        refreshSessions()
        return data
      } catch (jsonError) {
        const error = jsonError.status ? jsonError : streamError
        setApiOnline(false)
        updateMessage(assistantId, {
          heading: '',
          content: safeAgentErrorMessage(error),
          time,
          sources: [],
          citations: [],
          elapsedMs: performance.now() - startedAt,
          status: 'error',
          pending: false,
          progressSteps: [],
        })
        return null
      }
    } finally {
      setIsSending(false)
    }
  }

  async function copyAnswer(text) {
    await navigator.clipboard?.writeText(text)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1400)
  }

  async function sendFeedback(message, rating) {
    if (!message.id) return
    await apiFetch(`/messages/${message.id}/feedback`, {
      method: 'POST',
      body: JSON.stringify({ rating }),
    })
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1400)
  }

  async function shareMessage(message) {
    if (!message.id) return
    const data = await apiFetch(`/messages/${message.id}/share`, { method: 'POST' })
    const shareUrl = `${window.location.origin}${data.url}`
    await navigator.clipboard?.writeText(shareUrl)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1400)
  }

  async function downloadSession() {
    if (!sessionId) return
    const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/export.txt`)
    if (!response.ok) return
    const text = await response.text()
    const blob = new Blob([text], { type: 'text/plain' })
    const url = window.URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `insight-copilot-${sessionId}.txt`
    document.body.appendChild(link)
    link.click()
    link.remove()
    window.URL.revokeObjectURL(url)
  }

  async function retryMessage(message) {
    if (!message.id || isSending) return
    setIsSending(true)
    try {
      const data = await apiFetch(`/messages/${message.id}/retry`, {
        method: 'POST',
        body: JSON.stringify({}),
      })
      setSessionId(data.session_id)
      await loadSession(data.session_id)
    } finally {
      setIsSending(false)
    }
  }

  return (
    <div className={`app-shell ${sidebarCollapsed ? 'is-sidebar-collapsed' : ''}`}>
      <aside className={`sidebar ${sidebarOpen ? 'is-open' : ''}`}>
        <div className="sidebar__top">
          <div className="sidebar-brand">
            <img src={logoUrl} alt="" />
            <span>Insight Copilot</span>
          </div>
          <button className="new-chat" type="button" onClick={startNewChat}>
            <SquarePen size={17} />
            New chat
          </button>
          <button className="icon-button sidebar-close" type="button" aria-label="Close sidebar" onClick={() => setSidebarOpen(false)}>
            <ChevronLeft size={19} />
          </button>
        </div>

        <div className="history-list">
          {sessions.length ? (
            sessions.map((session) => (
              <button
                key={session.id}
                type="button"
                className={session.id === sessionId ? 'is-active' : ''}
                onClick={() => loadSession(session.id)}
              >
                <span>{session.title}</span>
                <small>{session.message_count} messages</small>
              </button>
            ))
          ) : (
            <p>{apiOnline ? 'No saved chats yet.' : 'History appears when the API is online.'}</p>
          )}
        </div>

        <div className="sidebar__foot">
          <span className={`connection ${apiOnline ? 'is-online' : ''}`}>
            <Check size={14} />
            {apiOnline ? 'API online' : 'Local preview'}
          </span>
        </div>
      </aside>
      <button
        className="sidebar-edge-toggle"
        type="button"
        aria-label={sidebarCollapsed ? 'Open history' : 'Close history'}
        onClick={() => setSidebarCollapsed((value) => !value)}
      >
        {sidebarCollapsed ? <ChevronRight size={19} /> : <ChevronLeft size={19} />}
      </button>

      <main className="chat-shell">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            type="button"
            aria-label="Open sidebar"
            onClick={() => {
              setSidebarCollapsed(false)
              setSidebarOpen(true)
            }}
          >
            <Menu size={20} />
          </button>
          <div className="topbar__title">
            <img src={logoUrl} alt="" />
            <strong>Insight Copilot</strong>
            {activeDocument ? <span>{activeDocument.title}</span> : null}
          </div>
          <div className="topbar__actions">
            <select aria-label="Source" value={activeDoc} onChange={(event) => setActiveDoc(event.target.value)}>
              <option value="all">All sources</option>
              {documents.map((doc) => (
                <option key={doc.id} value={doc.id}>
                  {doc.title} ({sourceTypeLabel(doc)})
                </option>
              ))}
            </select>
          </div>
        </header>

        <section className="chat-main">
          <div className="conversation">
            <div className="messages" aria-live="polite">
              {messages.length ? (
                messages.map((message) => (
                  <ChatMessage
                    key={message.id}
                    message={message}
                    onCopy={copyAnswer}
                    onFeedback={sendFeedback}
                    onShare={shareMessage}
                    onDownload={downloadSession}
                    onRetry={retryMessage}
                    isSending={isSending}
                  />
                ))
              ) : (
                <div className="empty-chat">
                  <div className="empty-chat__mark">
                    <img src={logoUrl} alt="" />
                  </div>
                  <h1>Shell Conversational AI</h1>
                  <p>Generate faster insight from your documents.</p>
                  <SuggestedQuestions examples={starterExamples} disabled={isSending} onSelect={submitQuestion} />
                </div>
              )}

              {latestFollowUps.length > 0 && !isSending ? (
                <SuggestedQuestions examples={latestFollowUps} disabled={isSending} onSelect={submitQuestion} followUp />
              ) : null}
            </div>

            <form
              className="composer"
              onSubmit={(event) => {
                event.preventDefault()
                submitQuestion()
              }}
            >
              <textarea
                ref={inputRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault()
                    submitQuestion()
                  }
                }}
                placeholder="Ask a Question"
                rows={1}
                disabled={isSending}
              />
              <div className="composer__controls">
                <button className="send-button" type="submit" aria-label="Send question" disabled={isSending || !input.trim()}>
                  <Send size={17} />
                </button>
              </div>
            </form>
          </div>
        </section>

        <div className={`copy-toast ${copied ? 'is-visible' : ''}`}>Copied</div>
      </main>
    </div>
  )
}

/* demo code push */
