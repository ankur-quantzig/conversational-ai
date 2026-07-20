import React, { useContext, useEffect, useMemo, useRef, useState } from 'react'
import {
  ArrowDown,
  BotMessageSquare,
  BookOpen,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  Check,
  CircleUserRound,
  Copy,
  Download,
  Menu,
  Moon,
  MoreHorizontal,
  Pencil,
  Pin,
  PinOff,
  Printer,
  RotateCcw,
  Search,
  Send,
  Share2,
  Square,
  SquarePen,
  Sun,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  X,
} from 'lucide-react'
import logoUrl from './assets/shell-logo.png'

const chunkFiles = import.meta.glob('../output/chunks/*-chunks.jsonl', {
  eager: true,
  import: 'default',
  query: '?raw',
})

const starterExamples = [
  'What is the main objective of the Conversational AI POC?',
  'Why do the videos need both transcription and frame-by-frame OCR?',
  'How should the solution connect video content with related documents?',
  'What Databricks architecture was proposed for SharePoint, Auto Loader, and the Bronze-Silver-Gold flow?',
]

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? '/api'
const APP_NAME = 'Shell Conversational AI'
const MAX_QUESTION_LENGTH = 4000
const MAX_TITLE_LENGTH = 200
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
    diagram: null,
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
  const isVideo = isVideoSource(source)
  const timeLabel =
    source.start_time_label && source.end_time_label
      ? `${source.start_time_label} - ${source.end_time_label}`
      : null
  const pages = source.page_numbers?.length ? source.page_numbers : citation?.pages
  if (isVideo && timeLabel) return [timeLabel]
  if (pages?.length) return pages.map((page) => `p. ${page}`)
  return [`source ${index + 1}`]
}

function isVideoSource(source = {}) {
  return String(source.source_type ?? source.metadata?.source_type ?? '').toLowerCase() === 'video'
}

function numericTime(value) {
  const number = Number(value)
  return Number.isFinite(number) && number >= 0 ? number : null
}

function VideoClipPlayer({ source }) {
  const videoRef = useRef(null)
  const [hasPlaybackError, setHasPlaybackError] = useState(false)
  const [playbackMode, setPlaybackMode] = useState('clip')
  const isVideo = isVideoSource(source)
  const encodedDocId = source.doc_id ? encodeURIComponent(source.doc_id) : ''
  const start = numericTime(source.start_time ?? source.metadata?.start_time) ?? 0
  const end = numericTime(source.end_time ?? source.metadata?.end_time)
  const hasEnd = end !== null && end > start
  const startLabel = source.start_time_label ?? source.metadata?.start_time_label ?? formatVideoTime(start)
  const endLabel = source.end_time_label ?? source.metadata?.end_time_label ?? (hasEnd ? formatVideoTime(end) : '')
  const clipLabel = endLabel ? `${startLabel} - ${endLabel}` : `From ${startLabel}`
  const timeFragment = `#t=${start}${hasEnd ? `,${end}` : ''}`
  const clipUrl =
    encodedDocId && hasEnd
      ? `${API_BASE_URL}/media/video-clips/${encodedDocId}?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`
      : ''
  const fullVideoUrl = encodedDocId ? `${API_BASE_URL}/media/videos/${encodedDocId}${timeFragment}` : ''
  const videoUrl = playbackMode === 'clip' && clipUrl ? clipUrl : fullVideoUrl

  useEffect(() => {
    setPlaybackMode(clipUrl ? 'clip' : 'full')
    setHasPlaybackError(false)
  }, [clipUrl, fullVideoUrl])

  if (!isVideo || !source.doc_id) return null

  function handleLoadedMetadata(event) {
    if (playbackMode !== 'full') return
    if (start > 0 && Math.abs(event.currentTarget.currentTime - start) > 1) {
      event.currentTarget.currentTime = start
    }
  }

  function handleTimeUpdate(event) {
    if (playbackMode !== 'full') return
    if (!hasEnd) return
    if (event.currentTarget.currentTime >= end) {
      event.currentTarget.pause()
    }
  }

  function handlePlaybackError() {
    if (playbackMode === 'clip' && fullVideoUrl) {
      setPlaybackMode('full')
      setHasPlaybackError(false)
      return
    }
    setHasPlaybackError(true)
  }

  return (
    <div className="video-clip">
      <div className="video-clip__meta">
        <span>Relevant clip</span>
        <strong>{clipLabel}</strong>
      </div>
      <video
        key={videoUrl}
        ref={videoRef}
        controls
        preload="metadata"
        src={videoUrl}
        onLoadedMetadata={handleLoadedMetadata}
        onTimeUpdate={handleTimeUpdate}
        onError={handlePlaybackError}
      />
      {hasPlaybackError ? (
        <div className="video-clip__error" role="status">
          Video source is not available for playback yet.
        </div>
      ) : null}
    </div>
  )
}

function sessionBucket(session) {
  const raw = session.updated_at ?? session.created_at
  const date = raw ? new Date(raw) : null
  if (!date || Number.isNaN(date.getTime())) return 'Recents'
  const now = new Date()
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate())
  if (date >= startOfToday) return 'Today'
  const startOfYesterday = new Date(startOfToday)
  startOfYesterday.setDate(startOfYesterday.getDate() - 1)
  if (date >= startOfYesterday) return 'Yesterday'
  const weekAgo = new Date(startOfToday)
  weekAgo.setDate(weekAgo.getDate() - 7)
  if (date >= weekAgo) return 'Previous 7 days'
  return 'Older'
}

function relativeTime(value) {
  const date = value ? new Date(value) : null
  if (!date || Number.isNaN(date.getTime())) return ''
  const minutes = Math.floor((Date.now() - date.getTime()) / 60000)
  if (minutes < 1) return 'just now'
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days}d ago`
  return new Intl.DateTimeFormat([], { month: 'short', day: 'numeric' }).format(date)
}

function timeGreeting() {
  const hour = new Date().getHours()
  if (hour < 12) return 'Good morning'
  if (hour < 17) return 'Good afternoon'
  return 'Good evening'
}

function firstNameFrom(profile) {
  const raw = profile?.name || profile?.email || ''
  if (!raw) return ''
  const base = raw.includes('@') ? raw.split('@')[0] : raw
  const first = base.split(/[\s._-]+/)[0]
  return first ? first.charAt(0).toUpperCase() + first.slice(1) : ''
}

function formatVideoTime(seconds) {
  const totalSeconds = Math.max(0, Math.floor(Number(seconds) || 0))
  const minutes = Math.floor(totalSeconds / 60)
  const remainder = totalSeconds % 60
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}

function sourceDisplayContent(source = {}) {
  const content = String(source.content || '').trim()
  if (isVideoSource(source)) return ''
  return content
}

function sourceTypeLabel(source = {}) {
  return isVideoSource(source) ? 'Video' : 'PDF/Text'
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
    isVideoSource(source)
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
      {!isVideoSource(source) && source.section ? <h4>{source.section}</h4> : null}
      <VideoClipPlayer source={source} />
      {!isVideoSource(source) && citation?.quote ? <blockquote>{citation.quote}</blockquote> : null}
      {displayContent ? (
        <p>{displayContent}</p>
      ) : null}
    </article>
  )
}

const CitationContext = React.createContext(null)

function CitationMark({ n }) {
  const context = useContext(CitationContext)
  const target = context?.resolve(n)
  if (target === null || target === undefined) return <>[{n}]</>
  return (
    <sup className="cite-mark">
      <button
        type="button"
        aria-label={`Open source ${n}`}
        aria-expanded={context.openSource === target}
        onClick={() => context.onToggle(target)}
      >
        {n}
      </button>
    </sup>
  )
}

function safeHref(url) {
  try {
    const parsed = new URL(String(url), window.location.origin)
    return ['http:', 'https:', 'mailto:'].includes(parsed.protocol) ? parsed.href : null
  } catch {
    return null
  }
}

function renderInline(text, keyPrefix = 'inline') {
  const pattern = /(\*\*[^*]+\*\*|`[^`]+`|\[[^\]]+\]\([^)\s]+\)|~~[^~\n]+~~|\[\d{1,2}\]|\*[^*\n]+\*)/g
  return String(text)
    .split(pattern)
    .map((part, index) => {
      const key = `${keyPrefix}-${index}`
      if (!part) return null
      if (part.startsWith('**') && part.endsWith('**') && part.length > 4) {
        return <strong key={key}>{part.slice(2, -2)}</strong>
      }
      if (part.startsWith('`') && part.endsWith('`') && part.length > 2) {
        return <code key={key}>{part.slice(1, -1)}</code>
      }
      if (part.startsWith('~~') && part.endsWith('~~') && part.length > 4) {
        return <del key={key}>{part.slice(2, -2)}</del>
      }
      if (part.startsWith('[')) {
        const link = part.match(/^\[([^\]]+)\]\(([^)\s]+)\)$/)
        if (link) {
          const href = safeHref(link[2])
          if (!href) return <span key={key}>{link[1]}</span>
          return (
            <a key={key} href={href} target="_blank" rel="noopener noreferrer">
              {link[1]}
            </a>
          )
        }
        const marker = part.match(/^\[(\d{1,2})\]$/)
        if (marker) {
          return <CitationMark key={key} n={Number(marker[1])} />
        }
      }
      if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
        return <em key={key}>{part.slice(1, -1)}</em>
      }
      return part
    })
}

function CodeBlock({ code, language }) {
  const [copied, setCopied] = useState(false)

  async function copyCode() {
    await navigator.clipboard?.writeText(code)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1400)
  }

  return (
    <div className="code-block">
      <div className="code-block__bar">
        <span>{language || 'code'}</span>
        <button type="button" aria-label="Copy code" onClick={copyCode}>
          {copied ? <Check size={13} /> : <Copy size={13} />}
          {copied ? 'Copied' : 'Copy'}
        </button>
      </div>
      <pre>
        <code>{code}</code>
      </pre>
    </div>
  )
}

function StructuredContent({ text, children }) {
  const blocks = []
  let bullets = []
  let numbered = []
  let paragraph = []

  function flushParagraph(key) {
    if (!paragraph.length) return
    blocks.push(<p key={key}>{renderInline(paragraph.join(' '), key)}</p>)
    paragraph = []
  }

  function flushBullets(key) {
    if (!bullets.length) return
    blocks.push(
      <ul key={key}>
        {bullets.map((bullet, index) => (
          <li key={`${key}-${index}`}>{renderInline(bullet, `${key}-${index}`)}</li>
        ))}
      </ul>,
    )
    bullets = []
  }

  function flushNumbered(key) {
    if (!numbered.length) return
    blocks.push(
      <ol key={key}>
        {numbered.map((item, index) => (
          <li key={`${key}-${index}`}>{renderInline(item, `${key}-${index}`)}</li>
        ))}
      </ol>,
    )
    numbered = []
  }

  function flushAll(key) {
    flushParagraph(`p-${key}`)
    flushBullets(`b-${key}`)
    flushNumbered(`n-${key}`)
  }

  const lines = String(text || '').split('\n')
  let index = 0
  while (index < lines.length) {
    const trimmed = lines[index].trim()
    const fence = trimmed.match(/^```(\w+)?/)
    if (fence) {
      flushAll(index)
      const codeLines = []
      index += 1
      while (index < lines.length && !lines[index].trim().startsWith('```')) {
        codeLines.push(lines[index])
        index += 1
      }
      blocks.push(<CodeBlock key={`code-${index}`} code={codeLines.join('\n')} language={fence[1]} />)
      index += 1
      continue
    }
    if (!trimmed) {
      flushAll(index)
      index += 1
      continue
    }
    const heading = trimmed.match(/^(#{1,4})\s+(.+)$/)
    if (heading) {
      flushAll(index)
      const Tag = heading[1].length <= 2 ? 'h3' : 'h4'
      blocks.push(<Tag key={`h-${index}`}>{renderInline(heading[2], `h-${index}`)}</Tag>)
      index += 1
      continue
    }
    if (/^(-{3,}|_{3,}|\*{3,})$/.test(trimmed)) {
      flushAll(index)
      blocks.push(<hr key={`hr-${index}`} />)
      index += 1
      continue
    }
    if (trimmed.startsWith('>')) {
      flushAll(index)
      const quoteLines = []
      while (index < lines.length && lines[index].trim().startsWith('>')) {
        quoteLines.push(lines[index].trim().replace(/^>\s?/, ''))
        index += 1
      }
      blocks.push(<blockquote key={`q-${index}`}>{renderInline(quoteLines.join(' '), `q-${index}`)}</blockquote>)
      continue
    }
    const nextLine = index + 1 < lines.length ? lines[index + 1].trim() : ''
    const isTableStart =
      trimmed.includes('|') && nextLine.includes('-') && /^\|?[\s:-]+(\|[\s:-]+)+\|?$/.test(nextLine)
    if (isTableStart) {
      flushAll(index)
      const splitRow = (row) =>
        row
          .replace(/^\|/, '')
          .replace(/\|$/, '')
          .split('|')
          .map((cell) => cell.trim())
      const headerCells = splitRow(trimmed)
      const tableKey = `t-${index}`
      index += 2
      const bodyRows = []
      while (index < lines.length && lines[index].includes('|') && lines[index].trim()) {
        bodyRows.push(splitRow(lines[index].trim()))
        index += 1
      }
      blocks.push(
        <div className="md-table" key={tableKey}>
          <table>
            <thead>
              <tr>
                {headerCells.map((cell, cellIndex) => (
                  <th key={`${tableKey}-h-${cellIndex}`}>{renderInline(cell, `${tableKey}-h-${cellIndex}`)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {bodyRows.map((row, rowIndex) => (
                <tr key={`${tableKey}-r-${rowIndex}`}>
                  {headerCells.map((_, cellIndex) => (
                    <td key={`${tableKey}-r-${rowIndex}-${cellIndex}`}>
                      {renderInline(row[cellIndex] ?? '', `${tableKey}-r-${rowIndex}-${cellIndex}`)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>,
      )
      continue
    }
    const bullet = trimmed.match(/^[-*]\s+(.+)$/)
    const ordered = trimmed.match(/^\d+[.)]\s+(.+)$/)
    if (bullet) {
      flushParagraph(`p-${index}`)
      flushNumbered(`n-${index}`)
      bullets.push(bullet[1])
    } else if (ordered) {
      flushParagraph(`p-${index}`)
      flushBullets(`b-${index}`)
      numbered.push(ordered[1])
    } else {
      flushBullets(`b-${index}`)
      flushNumbered(`n-${index}`)
      paragraph.push(trimmed)
    }
    index += 1
  }
  flushAll('last')

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

function ChatMessage({ message, onCopy, onFeedback, onShare, onDownload, onRetry, onEdit, isSending, isLast }) {
  const isUser = message.role === 'user'
  const Icon = isUser ? CircleUserRound : BotMessageSquare
  const [openSource, setOpenSource] = useState(null)
  const [showSources, setShowSources] = useState(false)
  const [justCopied, setJustCopied] = useState(false)
  const [isEditing, setIsEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const [collapsed, setCollapsed] = useState(false)
  const [collapsible, setCollapsible] = useState(false)
  const bubbleRef = useRef(null)

  useEffect(() => {
    if (isUser || message.pending || isLast || collapsible) return
    const el = bubbleRef.current
    if (el && el.scrollHeight > 560) {
      setCollapsible(true)
      setCollapsed(true)
    }
  }, [isUser, message.pending, isLast, collapsible])
  const citations = !isUser ? buildDisplayCitations(message) : []
  const activeCitation = openSource === null ? null : citations[openSource]
  const stageLabel = {
    getting_ready: 'Getting ready',
    thinking: 'Thinking',
    understanding: 'Understanding',
    finding_information: 'Finding information',
    reviewing_information: 'Reviewing information',
    preparing_answer: 'Preparing answer',
    finishing_up: 'Finishing up',
    working: 'Working',
    complete: 'Complete',
    writing: 'Writing',
    final_response: 'Final response',
    error: 'Needs attention',
  }[message.status]
  const actionsDisabled = message.pending || message.status === 'error'
  const toggleSource = (index) => setOpenSource(openSource === index ? null : index)

  function submitEdit() {
    const text = draft.trim()
    if (!text || isSending) return
    setIsEditing(false)
    onEdit(text)
  }
  const citationContext = {
    openSource,
    onToggle: toggleSource,
    resolve(n) {
      if (isUser || !citations.length) return null
      const byIndex = citations.findIndex((item) => item.citation?.source_index === n)
      if (byIndex >= 0) return byIndex
      return n >= 1 && n <= citations.length ? n - 1 : null
    },
  }

  const showStatusRow = !isUser && stageLabel && (message.pending || message.status === 'error')

  return (
    <article
      className={`message ${isUser ? 'message--user' : 'message--assistant'}`}
      aria-busy={!isUser && message.pending ? true : undefined}
    >
      {!isUser ? (
        <div className="avatar" aria-hidden="true">
          <Icon size={17} />
        </div>
      ) : null}
      <div className="message__stack">
        {showStatusRow ? (
          <div className="message__top">
            <span className={`stage-pill stage-pill--${message.status}`}>{stageLabel}</span>
            {message.elapsedMs ? <span className="response-time">{formatElapsed(message.elapsedMs)}</span> : null}
          </div>
        ) : null}
        {isUser && isEditing ? (
          <div className="message-edit">
            <textarea
              aria-label="Edit question"
              value={draft}
              maxLength={MAX_QUESTION_LENGTH}
              autoFocus
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault()
                  submitEdit()
                }
                if (event.key === 'Escape') {
                  event.stopPropagation()
                  setIsEditing(false)
                }
              }}
            />
            <div className="message-edit__actions">
              <span className="message-edit__hint">Enter to send · Esc to cancel</span>
              <button type="button" className="message-edit__cancel" onClick={() => setIsEditing(false)}>
                Cancel
              </button>
              <button type="button" className="message-edit__send" disabled={!draft.trim() || isSending} onClick={submitEdit}>
                Send
              </button>
            </div>
          </div>
        ) : (
          <div
            ref={bubbleRef}
            className={`bubble ${isUser ? 'bubble--user' : 'bubble--assistant'} ${collapsible && collapsed ? 'is-collapsed' : ''}`}
          >
            {!isUser && message.heading && !message.pending ? <h2 className="response-heading">{message.heading}</h2> : null}
            {!isUser && message.pending && !message.content && message.progressSteps?.length ? (
              <AgentStatusLine steps={message.progressSteps} />
            ) : null}
            {message.content ? (
              <CitationContext.Provider value={citationContext}>
                <StructuredContent text={message.content}>
                  {!isUser && message.pending ? <span className="stream-caret" aria-hidden="true" /> : null}
                  <CitationList items={citations} openSource={openSource} onToggle={toggleSource} />
                </StructuredContent>
              </CitationContext.Provider>
            ) : !isUser && message.pending && !message.progressSteps?.length ? (
              <div className="typing-indicator" role="status" aria-label="Agent is responding">
                <span />
                <span />
                <span />
              </div>
            ) : null}
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
          </div>
        )}
        {!isUser && collapsible ? (
          <button type="button" className="bubble-toggle" aria-expanded={!collapsed} onClick={() => setCollapsed((value) => !value)}>
            {collapsed ? 'Show more' : 'Show less'}
          </button>
        ) : null}
        {!isUser && !message.pending ? (
          <div className="message__meta">
            <div className="message-actions" aria-label="Message actions">
              <button
                type="button"
                aria-label="Copy response"
                title="Copy response"
                disabled={actionsDisabled}
                onClick={async () => {
                  await onCopy(message.content)
                  setJustCopied(true)
                  window.setTimeout(() => setJustCopied(false), 1400)
                }}
              >
                {justCopied ? <Check size={15} /> : <Copy size={15} />}
              </button>
              <button type="button" aria-label="Share response" title="Share response" disabled={actionsDisabled || !message.id} onClick={() => onShare(message)}>
                <Share2 size={15} />
              </button>
              <button type="button" aria-label="Download conversation" title="Download conversation" disabled={message.pending} onClick={() => onDownload()}>
                <Download size={15} />
              </button>
              <button
                type="button"
                className={message.feedback === 'up' ? 'is-selected' : ''}
                aria-label="Good response"
                aria-pressed={message.feedback === 'up'}
                title="Good response"
                disabled={actionsDisabled || !message.id}
                onClick={() => onFeedback(message, 'up')}
              >
                <ThumbsUp size={15} />
              </button>
              <button
                type="button"
                className={message.feedback === 'down' ? 'is-selected' : ''}
                aria-label="Bad response"
                aria-pressed={message.feedback === 'down'}
                title="Bad response"
                disabled={actionsDisabled || !message.id}
                onClick={() => onFeedback(message, 'down')}
              >
                <ThumbsDown size={15} />
              </button>
              <button type="button" aria-label="Try again" title="Try again" disabled={actionsDisabled || !message.id || isSending} onClick={() => onRetry(message)}>
                <RotateCcw size={15} />
              </button>
              <button type="button" aria-label="Show sources" title="Show sources" disabled={actionsDisabled || !citations.length} onClick={() => setShowSources((value) => !value)}>
                <BookOpen size={15} />
              </button>
            </div>
            {message.elapsedMs ? <span className="response-time">{formatElapsed(message.elapsedMs)}</span> : null}
            {message.time ? <time>{message.time}</time> : null}
          </div>
        ) : null}
        {isUser && !isEditing ? (
          <div className="message__meta">
            <div className="message-actions message-actions--user" aria-label="Question actions">
              <button
                type="button"
                aria-label="Edit question"
                title="Edit and resend"
                disabled={isSending}
                onClick={() => {
                  setDraft(message.content)
                  setIsEditing(true)
                }}
              >
                <Pencil size={14} />
              </button>
              <button
                type="button"
                aria-label="Copy question"
                title="Copy question"
                onClick={async () => {
                  await onCopy(message.content)
                  setJustCopied(true)
                  window.setTimeout(() => setJustCopied(false), 1400)
                }}
              >
                {justCopied ? <Check size={14} /> : <Copy size={14} />}
              </button>
            </div>
            {message.time ? <time>{message.time}</time> : null}
          </div>
        ) : null}
      </div>
    </article>
  )
}

function AgentStatusLine({ steps }) {
  const [open, setOpen] = useState(false)
  const current = steps[steps.length - 1]
  return (
    <div className="agent-status" aria-label="Agent progress">
      <button
        type="button"
        className="agent-status__line"
        aria-expanded={open}
        title={open ? 'Hide steps' : 'Show all steps'}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="agent-status__dot" aria-hidden="true" />
        <span className="agent-status__text">{current.message}…</span>
        <ChevronDown size={13} className={open ? 'is-open' : ''} aria-hidden="true" />
      </button>
      {open ? (
        <div className="agent-status__steps">
          {steps.map((step, index) => (
            <span key={`${step.stage}-${index}`} className={index === steps.length - 1 ? 'is-active' : ''}>
              {step.message}
            </span>
          ))}
        </div>
      ) : null}
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
  const [me, setMe] = useState(null)
  const [isSending, setIsSending] = useState(false)
  const [toast, setToast] = useState({ text: '', kind: 'default', visible: false })
  const [announcement, setAnnouncement] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [atBottom, setAtBottom] = useState(true)
  const [pinnedIds, setPinnedIds] = useState(() => {
    try {
      const stored = JSON.parse(window.localStorage.getItem('insight-copilot-pinned') ?? '[]')
      return Array.isArray(stored) ? stored : []
    } catch {
      return []
    }
  })
  const [menuSessionId, setMenuSessionId] = useState(null)
  const [renamingSessionId, setRenamingSessionId] = useState(null)
  const [renameValue, setRenameValue] = useState('')
  const [historyQuery, setHistoryQuery] = useState('')
  const [theme, setTheme] = useState(() => {
    const urlTheme = new URLSearchParams(window.location.search).get('theme')
    if (urlTheme === 'dark' || urlTheme === 'light') return urlTheme
    const stored = window.localStorage.getItem('insight-copilot-theme')
    if (stored === 'dark' || stored === 'light') return stored
    return window.matchMedia?.('(prefers-color-scheme: dark)')?.matches ? 'dark' : 'light'
  })
  const inputRef = useRef(null)
  const messagesRef = useRef(null)
  const toastTimer = useRef(null)
  const abortRef = useRef(null)
  const mobileMenuRef = useRef(null)
  const sidebarCloseRef = useRef(null)
  const menuTriggerRef = useRef(null)
  const prevSidebarOpen = useRef(false)

  const documents = apiDocuments.length ? apiDocuments : localDocuments
  const activeDocument = documents.find((doc) => doc.id === activeDoc)
  const historyFilter = historyQuery.trim().toLowerCase()
  const filteredSessions = historyFilter
    ? sessions.filter((session) => String(session.title ?? '').toLowerCase().includes(historyFilter))
    : sessions
  const pinnedSessions = filteredSessions.filter((session) => pinnedIds.includes(session.id))
  const recentSessions = filteredSessions.filter((session) => !pinnedIds.includes(session.id))
  const sessionGroups = []
  recentSessions.forEach((session) => {
    const label = sessionBucket(session)
    let group = sessionGroups.find((entry) => entry.label === label)
    if (!group) {
      group = { label, sessions: [] }
      sessionGroups.push(group)
    }
    group.sessions.push(session)
  })
  const latestFollowUps =
    [...messages]
      .reverse()
      .find((message) => message.role === 'assistant' && !message.pending && message.followUpQuestions?.length)
      ?.followUpQuestions ?? []

  useEffect(() => {
    refreshSessions()
    inputRef.current?.focus()
  }, [])

  useEffect(() => {
    if (!atBottom) return
    const el = messagesRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [messages, atBottom])

  useEffect(() => {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }, [input])

  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === 'Escape') {
        setSidebarOpen(false)
        setMenuSessionId(null)
        setRenamingSessionId(null)
        menuTriggerRef.current?.focus()
        menuTriggerRef.current = null
        return
      }
      if ((event.ctrlKey || event.metaKey) && event.shiftKey && event.key.toLowerCase() === 'o') {
        event.preventDefault()
        setSessionId(null)
        setMessages([])
        setSidebarOpen(false)
        inputRef.current?.focus()
        return
      }
      if (event.key === '/' && !event.ctrlKey && !event.metaKey && !event.altKey) {
        const target = event.target
        const tag = target?.tagName
        if (tag !== 'INPUT' && tag !== 'TEXTAREA' && !target?.isContentEditable) {
          event.preventDefault()
          inputRef.current?.focus()
        }
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  useEffect(() => {
    window.localStorage.setItem('insight-copilot-pinned', JSON.stringify(pinnedIds))
  }, [pinnedIds])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    window.localStorage.setItem('insight-copilot-theme', theme)
  }, [theme])

  const wasSending = useRef(false)
  useEffect(() => {
    if (isSending) {
      setAnnouncement('Generating answer')
    } else if (wasSending.current) {
      setAnnouncement('Response ready')
    }
    wasSending.current = isSending
  }, [isSending])

  useEffect(() => {
    if (!menuSessionId) return
    document.querySelector('.history-menu button')?.focus()
    function handleClick() {
      setMenuSessionId(null)
    }
    window.addEventListener('click', handleClick)
    return () => window.removeEventListener('click', handleClick)
  }, [menuSessionId])

  useEffect(() => {
    if (sidebarOpen) {
      sidebarCloseRef.current?.focus()
    } else if (prevSidebarOpen.current) {
      mobileMenuRef.current?.focus()
    }
    prevSidebarOpen.current = sidebarOpen
  }, [sidebarOpen])

  function showToast(text, kind = 'default') {
    setToast({ text, kind, visible: true })
    window.clearTimeout(toastTimer.current)
    toastTimer.current = window.setTimeout(() => setToast((current) => ({ ...current, visible: false })), 1800)
  }

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
      const [sessionData, documentData, meData] = await Promise.all([
        apiFetch('/sessions'),
        apiFetch('/documents'),
        apiFetch('/me').catch(() => null),
      ])
      setSessions(sessionData)
      setApiDocuments(documentData)
      setMe(meData)
      setApiOnline(true)
    } catch {
      setApiOnline(false)
      setMe(null)
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

  function togglePin(targetSessionId) {
    setPinnedIds((current) =>
      current.includes(targetSessionId) ? current.filter((id) => id !== targetSessionId) : [targetSessionId, ...current],
    )
    setMenuSessionId(null)
  }

  function startRenaming(session) {
    setRenamingSessionId(session.id)
    setRenameValue(session.title)
    setMenuSessionId(null)
  }

  async function renameSession(targetSessionId) {
    const title = renameValue.trim()
    setRenamingSessionId(null)
    const existing = sessions.find((session) => session.id === targetSessionId)
    if (!title || !existing || title === existing.title) return
    try {
      await apiFetch(`/sessions/${targetSessionId}`, {
        method: 'PATCH',
        body: JSON.stringify({ title }),
      })
      setSessions((current) => current.map((session) => (session.id === targetSessionId ? { ...session, title } : session)))
      showToast('Chat renamed')
    } catch {
      showToast('Could not rename chat', 'error')
    }
  }

  async function deleteSession(targetSessionId) {
    setMenuSessionId(null)
    if (!window.confirm('Delete this chat? This cannot be undone.')) return
    try {
      await apiFetch(`/sessions/${targetSessionId}`, { method: 'DELETE' })
      setSessions((current) => current.filter((session) => session.id !== targetSessionId))
      setPinnedIds((current) => current.filter((id) => id !== targetSessionId))
      if (targetSessionId === sessionId) startNewChat()
      showToast('Chat deleted')
    } catch {
      showToast('Could not delete chat', 'error')
    }
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
      diagram: null,
      questionAnalysis: data.question_analysis ?? null,
      elapsedMs,
      status: 'final_response',
      pending: false,
      progressSteps: [],
    })
    return data
  }

  async function submitChatStream(payload, assistantId, startedAt, time) {
    const controller = new AbortController()
    abortRef.current = controller
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
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
          diagram: null,
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
    const question = questionText.trim().slice(0, MAX_QUESTION_LENGTH)
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
        status: 'getting_ready',
        pending: true,
        progressSteps: [{ stage: 'getting_ready', message: 'Getting ready', metadata: {} }],
      },
    ])
    setInput('')
    setIsSending(true)
    setAtBottom(true)

    try {
      const streamed = await submitChatStream(payload, assistantId, startedAt, time)
      if (streamed) refreshSessions()
    } catch (streamError) {
      if (streamError.name === 'AbortError') {
        setMessages((current) =>
          current.map((message) =>
            message.id === assistantId
              ? {
                  ...message,
                  content: message.content || 'Response generation was stopped.',
                  status: 'final_response',
                  pending: false,
                  streamingStarted: false,
                  progressSteps: [],
                }
              : message,
          ),
        )
        return null
      }
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
      abortRef.current = null
      setIsSending(false)
    }
  }

  async function copyAnswer(text) {
    await navigator.clipboard?.writeText(text)
    showToast('Copied to clipboard')
  }

  async function sendFeedback(message, rating) {
    if (!message.id) return
    try {
      await apiFetch(`/messages/${message.id}/feedback`, {
        method: 'POST',
        body: JSON.stringify({ rating }),
      })
      updateMessage(message.id, { feedback: rating })
      showToast('Feedback sent')
    } catch {
      showToast('Could not send feedback', 'error')
    }
  }

  async function shareMessage(message) {
    if (!message.id) return
    try {
      const data = await apiFetch(`/messages/${message.id}/share`, { method: 'POST' })
      const shareUrl = `${window.location.origin}${data.url}`
      await navigator.clipboard?.writeText(shareUrl)
      showToast('Share link copied')
    } catch {
      showToast('Could not create share link', 'error')
    }
  }

  async function downloadSession() {
    if (!sessionId) return
    const response = await fetch(`${API_BASE_URL}/sessions/${sessionId}/export.txt`)
    if (!response.ok) {
      showToast('Could not download transcript', 'error')
      return
    }
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
    } catch {
      showToast('Could not retry this message', 'error')
    } finally {
      setIsSending(false)
    }
  }

  function renderSessionRow(session) {
    const isPinned = pinnedIds.includes(session.id)
    if (renamingSessionId === session.id) {
      return (
        <input
          key={session.id}
          className="history-rename"
          aria-label="Rename chat"
          value={renameValue}
          maxLength={MAX_TITLE_LENGTH}
          autoFocus
          onFocus={(event) => event.currentTarget.select()}
          onChange={(event) => setRenameValue(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault()
              event.currentTarget.blur()
            }
          }}
          onBlur={() => renameSession(session.id)}
        />
      )
    }
    return (
      <div className={`history-row ${session.id === sessionId ? 'is-active' : ''}`} key={session.id}>
        <button
          type="button"
          className="history-row__open"
          aria-current={session.id === sessionId ? 'true' : undefined}
          onClick={() => loadSession(session.id)}
        >
          <span className="history-title">
            {isPinned ? <Pin size={12} /> : null}
            <span className="history-title__text">{session.title}</span>
          </span>
          <small>
            {session.message_count} messages
            {relativeTime(session.updated_at ?? session.created_at) ? ` · ${relativeTime(session.updated_at ?? session.created_at)}` : ''}
          </small>
        </button>
        <button
          type="button"
          className="history-row__menu"
          aria-label={`Options for ${session.title}`}
          aria-haspopup="menu"
          aria-expanded={menuSessionId === session.id}
          onClick={(event) => {
            event.stopPropagation()
            menuTriggerRef.current = event.currentTarget
            setMenuSessionId(menuSessionId === session.id ? null : session.id)
          }}
        >
          <MoreHorizontal size={16} />
        </button>
        {menuSessionId === session.id ? (
          <div
            className="history-menu"
            role="menu"
            aria-label={`Options for ${session.title}`}
            onClick={(event) => event.stopPropagation()}
            onKeyDown={(event) => {
              if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
              event.preventDefault()
              const items = [...event.currentTarget.querySelectorAll('button')]
              const index = items.indexOf(document.activeElement)
              const nextIndex =
                event.key === 'ArrowDown' ? (index + 1) % items.length : (index - 1 + items.length) % items.length
              items[nextIndex]?.focus()
            }}
          >
            <button type="button" role="menuitem" onClick={() => togglePin(session.id)}>
              {isPinned ? <PinOff size={14} /> : <Pin size={14} />}
              {isPinned ? 'Unpin' : 'Pin'}
            </button>
            <button type="button" role="menuitem" onClick={() => startRenaming(session)}>
              <Pencil size={14} />
              Rename
            </button>
            <button type="button" role="menuitem" className="is-danger" onClick={() => deleteSession(session.id)}>
              <Trash2 size={14} />
              Delete
            </button>
          </div>
        ) : null}
      </div>
    )
  }

  return (
    <div className={`app-shell ${sidebarCollapsed ? 'is-sidebar-collapsed' : ''}`}>
      <a className="skip-link" href="#composer-input">
        Skip to question input
      </a>
      <aside className={`sidebar ${sidebarOpen ? 'is-open' : ''}`} aria-label="Chat history">
        <div className="sidebar__top">
          <div className="sidebar-brand">
            <img src={logoUrl} alt="" />
            <span>{timeGreeting()}{firstNameFrom(me) ? `, ${firstNameFrom(me)}` : ''}</span>
          </div>
          <button className="new-chat" type="button" onClick={startNewChat}>
            <SquarePen size={17} />
            New chat
          </button>
          <button ref={sidebarCloseRef} className="icon-button sidebar-close" type="button" aria-label="Close sidebar" onClick={() => setSidebarOpen(false)}>
            <ChevronLeft size={19} />
          </button>
          {sessions.length ? (
            <div className="history-search">
              <Search size={14} aria-hidden="true" />
              <input
                type="search"
                value={historyQuery}
                maxLength={120}
                placeholder="Search chats"
                aria-label="Search chats"
                onChange={(event) => setHistoryQuery(event.target.value)}
              />
              {historyQuery ? (
                <button type="button" aria-label="Clear search" onClick={() => setHistoryQuery('')}>
                  <X size={13} />
                </button>
              ) : null}
            </div>
          ) : null}
        </div>

        <div className="history-list">
          {filteredSessions.length ? (
            <>
              {pinnedSessions.length ? <p className="history-label">Pinned</p> : null}
              {pinnedSessions.map(renderSessionRow)}
              {sessionGroups.map((group) => (
                <React.Fragment key={group.label}>
                  <p className="history-label">{group.label}</p>
                  {group.sessions.map(renderSessionRow)}
                </React.Fragment>
              ))}
            </>
          ) : (
            <p>
              {sessions.length
                ? 'No chats match your search.'
                : apiOnline
                  ? 'No saved chats yet.'
                  : 'History appears when the API is online.'}
            </p>
          )}
        </div>

        <div className="sidebar__foot">
          {me?.email ? (
            <div className="user-badge">
              <span className="user-badge__avatar" aria-hidden="true">
                {(firstNameFrom(me) || me.email).charAt(0).toUpperCase()}
              </span>
              <div className="user-badge__meta">
                <span className="user-badge__email">{me.email}</span>
              </div>
              <span
                className={`status-dot ${apiOnline ? 'is-online' : ''}`}
                role="status"
                title={apiOnline ? 'API online' : 'API offline'}
                aria-label={apiOnline ? 'API online' : 'API offline'}
              />
            </div>
          ) : (
            <span className={`connection ${apiOnline ? 'is-online' : ''}`}>
              <Check size={14} />
              {apiOnline ? 'API online' : 'Local preview'}
            </span>
          )}
        </div>
      </aside>
      {sidebarOpen ? <div className="sidebar-backdrop" aria-hidden="true" onClick={() => setSidebarOpen(false)} /> : null}
      <button
        className="sidebar-edge-toggle"
        type="button"
        aria-expanded={!sidebarCollapsed}
        aria-label={sidebarCollapsed ? 'Open history' : 'Close history'}
        onClick={() => setSidebarCollapsed((value) => !value)}
      >
        {sidebarCollapsed ? <ChevronRight size={19} /> : <ChevronLeft size={19} />}
      </button>

      <main className="chat-shell">
        <header className="topbar">
          <button
            ref={mobileMenuRef}
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
          <div className="topbar__actions">
            {messages.length ? (
              <button
                className="icon-button"
                type="button"
                aria-label="Print or save conversation as PDF"
                title="Print / save as PDF"
                onClick={() => window.print()}
              >
                <Printer size={18} />
              </button>
            ) : null}
            <button
              className="icon-button"
              type="button"
              aria-label="Toggle theme"
              title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            >
              {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
            </button>
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
            <div
              className="messages"
              role="log"
              aria-live="off"
              aria-label="Conversation"
              ref={messagesRef}
              onScroll={(event) => {
                const el = event.currentTarget
                setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 120)
              }}
            >
              {messages.length ? (
                messages.map((message, index) => (
                  <ChatMessage
                    key={message.id}
                    message={message}
                    isLast={index === messages.length - 1}
                    onCopy={copyAnswer}
                    onFeedback={sendFeedback}
                    onShare={shareMessage}
                    onDownload={downloadSession}
                    onRetry={retryMessage}
                    onEdit={(text) => submitQuestion(text)}
                    isSending={isSending}
                  />
                ))
              ) : (
                <div className="empty-chat">
                  <div className="empty-chat__mark">
                    <img src={logoUrl} alt="" />
                  </div>
                  <h1>{APP_NAME}</h1>
                  <p>Ask anything about your indexed documents and videos.</p>
                  <SuggestedQuestions examples={starterExamples} disabled={isSending} onSelect={submitQuestion} />
                </div>
              )}

              {latestFollowUps.length > 0 && !isSending ? (
                <SuggestedQuestions examples={latestFollowUps} disabled={isSending} onSelect={submitQuestion} followUp />
              ) : null}
            </div>

            {!atBottom && messages.length ? (
              <button
                className="scroll-bottom"
                type="button"
                aria-label="Scroll to latest message"
                title="Scroll to latest"
                onClick={() => {
                  const el = messagesRef.current
                  el?.scrollTo({ top: el.scrollHeight, behavior: 'smooth' })
                  setAtBottom(true)
                }}
              >
                <ArrowDown size={16} />
              </button>
            ) : null}

            <form
              className="composer"
              onSubmit={(event) => {
                event.preventDefault()
                submitQuestion()
              }}
            >
              {activeDoc !== 'all' && activeDocument ? (
                <div className="composer__filter">
                  <span>
                    Searching in: <strong>{activeDocument.title}</strong>
                  </span>
                  <button type="button" aria-label="Clear source filter" title="Search all sources" onClick={() => setActiveDoc('all')}>
                    <X size={13} />
                  </button>
                </div>
              ) : null}
              <textarea
                id="composer-input"
                aria-label="Ask a question"
                ref={inputRef}
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault()
                    submitQuestion()
                  }
                }}
                placeholder="Ask a question…"
                rows={1}
                maxLength={MAX_QUESTION_LENGTH}
              />
              <div className="composer__controls">
                <span className="composer__hint">Enter to send · Shift+Enter for a new line</span>
                {input.length >= MAX_QUESTION_LENGTH - 400 ? (
                  <span className="composer__count">
                    {input.length}/{MAX_QUESTION_LENGTH}
                  </span>
                ) : null}
                {isSending ? (
                  <button
                    className="send-button send-button--stop"
                    type="button"
                    aria-label="Stop generating"
                    title="Stop generating"
                    onClick={() => abortRef.current?.abort()}
                  >
                    <Square size={13} fill="currentColor" />
                  </button>
                ) : (
                  <button className="send-button" type="submit" aria-label="Send question" disabled={!input.trim()}>
                    <Send size={17} />
                  </button>
                )}
              </div>
            </form>
          </div>
        </section>

        <div
          className={`copy-toast copy-toast--${toast.kind} ${toast.visible ? 'is-visible' : ''}`}
          role={toast.kind === 'error' ? 'alert' : 'status'}
        >
          {toast.text}
        </div>
        <div className="sr-only" aria-live="polite">
          {announcement}
        </div>
      </main>
    </div>
  )
}

/* demo code push */
