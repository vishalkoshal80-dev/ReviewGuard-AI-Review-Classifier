import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertCircle,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  BarChart3,
  Bell,
  Check,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  Clock3,
  CloudUpload,
  Cpu,
  Database,
  Download,
  FileSpreadsheet,
  Fingerprint,
  Gauge,
  Github,
  LayoutDashboard,
  LoaderCircle,
  Menu,
  MessageSquareText,
  Microscope,
  Network,
  PanelLeftClose,
  Search,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Upload,
  WandSparkles,
  X,
  Zap,
} from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000'

const navigation = [
  { id: 'Overview', label: 'Overview', icon: LayoutDashboard, group: 'WORKSPACE' },
  { id: 'Analyze review', label: 'Analyze review', icon: MessageSquareText, group: 'WORKSPACE' },
  { id: 'Batch analysis', label: 'Batch analysis', icon: FileSpreadsheet, group: 'WORKSPACE' },
  { id: 'Model performance', label: 'Model performance', icon: BarChart3, group: 'INTELLIGENCE' },
]

const sampleReviews = [
  {
    label: 'Product praise',
    text: 'Love this! Well made, sturdy, and very comfortable. I love it! Very pretty',
  },
  {
    label: 'Balanced feedback',
    text: 'The headphones work well. Battery life is decent, and the sound quality is acceptable for the price.',
  },
  {
    label: 'Critical review',
    text: 'The item arrived late and the packaging was damaged. The product works, but the quality was lower than expected.',
  },
]

function formatPercent(value) {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : '—'
}

function formatNumber(value) {
  return typeof value === 'number' ? value.toLocaleString() : '—'
}

function toCsv(rows) {
  if (!rows?.length) return ''
  const columns = [...new Set(rows.flatMap((row) => Object.keys(row)))]
  const escapeCell = (value) => {
    const text = value == null ? '' : String(value)
    return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text
  }
  return [columns.map(escapeCell).join(','), ...rows.map((row) => columns.map((column) => escapeCell(row[column])).join(','))].join('\r\n')
}

function MetricCard({ label, value, sublabel, icon: Icon, accent = 'cyan', trend }) {
  return (
    <article className={`metric-card accent-${accent}`}>
      <div className="metric-card-top">
        <span className="metric-label">{label}</span>
        <span className="metric-icon"><Icon size={18} strokeWidth={1.8} /></span>
      </div>
      <div className="metric-value-row">
        <strong className="metric-value">{value}</strong>
        {trend && <span className="metric-trend"><ArrowUpRight size={13} /> {trend}</span>}
      </div>
      <p className="metric-sublabel">{sublabel}</p>
    </article>
  )
}

function SectionHeading({ eyebrow, title, description, action }) {
  return (
    <div className="section-heading">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h2>{title}</h2>
        {description && <p>{description}</p>}
      </div>
      {action}
    </div>
  )
}

function App() {
  const [page, setPage] = useState('Overview')
  const [metrics, setMetrics] = useState(null)
  const [apiStatus, setApiStatus] = useState('checking')
  const [loadError, setLoadError] = useState('')
  const [reviewText, setReviewText] = useState('')
  const [reviewResult, setReviewResult] = useState(null)
  const [predicting, setPredicting] = useState(false)
  const [predictError, setPredictError] = useState('')
  const [batchFile, setBatchFile] = useState(null)
  const [textColumn, setTextColumn] = useState('auto')
  const [batchResult, setBatchResult] = useState(null)
  const [batching, setBatching] = useState(false)
  const [batchError, setBatchError] = useState('')
  const [mobileSidebar, setMobileSidebar] = useState(false)
  const [dragActive, setDragActive] = useState(false)
  const [notice, setNotice] = useState('')

  const fetchDashboard = useCallback(async () => {
    try {
      const [healthResponse, metricsResponse] = await Promise.all([
        fetch(`${API_BASE}/api/health`),
        fetch(`${API_BASE}/api/metrics`),
      ])
      if (!healthResponse.ok) throw new Error('API did not respond. Start the FastAPI backend.')
      const health = await healthResponse.json()
      const metricData = metricsResponse.ok ? await metricsResponse.json() : {}
      setApiStatus(health.status === 'ok' && health.model_file_exists ? 'online' : 'warning')
      setMetrics(metricData)
      setLoadError(health.model_file_exists ? '' : 'The backend is running, but the saved model file was not found.')
    } catch (error) {
      setApiStatus('offline')
      setLoadError(error.message || 'Could not connect to the backend.')
    }
  }, [])

  useEffect(() => {
    fetchDashboard()
  }, [fetchDashboard])

  const metricCards = useMemo(() => [
    {
      label: 'Test accuracy',
      value: formatPercent(metrics?.accuracy),
      sublabel: 'Held-out evaluation split',
      icon: Gauge,
      accent: 'cyan',
    },
    {
      label: 'Precision · CG',
      value: formatPercent(metrics?.precision_cg),
      sublabel: 'Positive predictive value',
      icon: Fingerprint,
      accent: 'purple',
    },
    {
      label: 'Recall · CG',
      value: formatPercent(metrics?.recall_cg),
      sublabel: 'CG examples found',
      icon: Microscope,
      accent: 'blue',
    },
    {
      label: 'F1-score · CG',
      value: formatPercent(metrics?.f1_cg),
      sublabel: 'Precision and recall balance',
      icon: Zap,
      accent: 'green',
    },
  ], [metrics])

  const goTo = (nextPage) => {
    setPage(nextPage)
    setMobileSidebar(false)
    setNotice('')
  }

  const analyzeReview = async (event) => {
    event?.preventDefault()
    setPredictError('')
    setReviewResult(null)
    if (!reviewText.trim()) {
      setPredictError('Add a review first, then run the analysis.')
      return
    }
    setPredicting(true)
    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: reviewText.trim() }),
      })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail || 'Prediction request failed.')
      setReviewResult(body)
    } catch (error) {
      setPredictError(error.message || 'Could not analyze this review.')
    } finally {
      setPredicting(false)
    }
  }

  const runBatch = async (event) => {
    event?.preventDefault()
    setBatchError('')
    setBatchResult(null)
    if (!batchFile) {
      setBatchError('Choose a CSV file to continue.')
      return
    }
    setBatching(true)
    const form = new FormData()
    form.append('file', batchFile)
    form.append('text_column', textColumn)
    try {
      const response = await fetch(`${API_BASE}/api/batch`, { method: 'POST', body: form })
      const body = await response.json()
      if (!response.ok) throw new Error(body.detail || 'Batch analysis failed.')
      setBatchResult(body)
    } catch (error) {
      setBatchError(error.message || 'Could not analyze this CSV.')
    } finally {
      setBatching(false)
    }
  }

  const downloadBatch = () => {
    if (!batchResult?.rows?.length) return
    const blob = new Blob([toCsv(batchResult.rows)], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'reviewguard_predictions.csv'
    document.body.appendChild(anchor)
    anchor.click()
    anchor.remove()
    URL.revokeObjectURL(url)
  }

  const fillSample = (text) => {
    setReviewText(text)
    setPage('Analyze review')
    setReviewResult(null)
    setPredictError('')
  }

  const renderSidebar = () => (
    <>
      <div className="sidebar-brand">
        <div className="brand-mark"><ShieldCheck size={21} strokeWidth={2.2} /></div>
        <div className="brand-copy">
          <div className="brand-name">review<span>guard</span></div>
          <div className="brand-caption">AI REVIEW INTELLIGENCE</div>
        </div>
        <button className="mobile-close icon-button" onClick={() => setMobileSidebar(false)} aria-label="Close navigation"><X size={18} /></button>
      </div>

      <div className="workspace-chip">
        <div className="workspace-avatar">RG</div>
        <div className="workspace-copy"><strong>ReviewGuard Lab</strong><span>Personal workspace</span></div>
        <ChevronDown size={15} className="muted-icon" />
      </div>

      {['WORKSPACE', 'INTELLIGENCE'].map((group) => (
        <div className="nav-group" key={group}>
          <div className="nav-group-label">{group}</div>
          {navigation.filter((item) => item.group === group).map((item) => {
            const Icon = item.icon
            return (
              <button key={item.id} className={`nav-item ${page === item.id ? 'active' : ''}`} onClick={() => goTo(item.id)}>
                <Icon size={18} strokeWidth={1.8} />
                <span>{item.label}</span>
                {item.id === 'Batch analysis' && <span className="nav-new">CSV</span>}
              </button>
            )
          })}
        </div>
      ))}

      <div className="sidebar-bottom">
        <div className="sidebar-api-card">
          <div className="sidebar-api-top">
            <span className={`status-dot ${apiStatus}`} />
            <strong>API status</strong>
            <span className={`api-status-text ${apiStatus}`}>{apiStatus === 'online' ? 'Connected' : apiStatus === 'checking' ? 'Checking' : 'Offline'}</span>
          </div>
          <p>{apiStatus === 'online' ? 'Model service is ready for inference.' : 'Start the backend to enable predictions.'}</p>
          <button className="text-button" onClick={fetchDashboard}>Refresh connection <ArrowRight size={13} /></button>
        </div>
        <div className="sidebar-user">
          <div className="user-avatar">V</div>
          <div className="user-copy"><strong>Project workspace</strong><span>Local development</span></div>
          <button className="icon-button small-icon" title="Workspace details" onClick={() => setNotice('ReviewGuard runs locally. Your review text is sent to your local FastAPI server.')}><CircleHelp size={17} /></button>
        </div>
      </div>
    </>
  )

  return (
    <div className="app-shell">
      {mobileSidebar && <button className="mobile-scrim" onClick={() => setMobileSidebar(false)} aria-label="Close menu overlay" />}
      <aside className={`sidebar ${mobileSidebar ? 'sidebar-open' : ''}`}>{renderSidebar()}</aside>

      <main className="main-area">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-button mobile-menu" onClick={() => setMobileSidebar(true)} aria-label="Open navigation"><Menu size={19} /></button>
            <div className="breadcrumb"><span>Workspace</span><span className="breadcrumb-slash">/</span><strong>{page}</strong></div>
          </div>
          <div className="topbar-right">
            <div className="local-badge"><span className={`status-dot ${apiStatus}`} /> Local environment</div>
            <button className="icon-button topbar-bell" title="System status" onClick={() => setNotice(loadError || 'All set. ReviewGuard is ready for local testing.')}><Bell size={18} /><span className="bell-dot" /></button>
            <div className="topbar-avatar">V</div>
          </div>
        </header>

        <div className="page-content">
          {notice && <div className="notice-banner"><CircleHelp size={16} /><span>{notice}</span><button className="icon-button" onClick={() => setNotice('')}><X size={15} /></button></div>}
          {loadError && <div className={`connection-banner ${apiStatus === 'offline' ? 'error' : ''}`}><AlertCircle size={17} /><div><strong>{apiStatus === 'offline' ? 'Backend not connected' : 'Model unavailable'}</strong><span>{loadError} Start FastAPI with <code>python -m uvicorn backend.main:app --reload --port 8000</code>.</span></div><button className="text-button" onClick={fetchDashboard}>Retry</button></div>}

          {page === 'Overview' && (
            <>
              <section className="welcome-row">
                <div>
                  <div className="eyebrow"><Sparkles size={13} /> INTELLIGENCE OVERVIEW</div>
                  <h1>Good evening, <span>Vishal.</span></h1>
                  <p className="page-description">Your review intelligence workspace, ready when you are.</p>
                </div>
                <button className="primary-button" onClick={() => goTo('Analyze review')}><WandSparkles size={17} /> New analysis <ArrowRight size={15} /></button>
              </section>

              <section className="metric-grid">
                {metricCards.map((card) => <MetricCard key={card.label} {...card} />)}
              </section>

              <section className="overview-grid">
                <article className="panel performance-panel">
                  <div className="panel-heading">
                    <div><div className="eyebrow">MODEL PERFORMANCE</div><h3>Evaluation snapshot</h3><p>Held-out test split · Logistic Regression</p></div>
                    <div className="panel-icon cyan-icon"><Activity size={19} /></div>
                  </div>
                  <div className="performance-main">
                    <div className="accuracy-ring" style={{ '--ring-value': `${(metrics?.accuracy || 0) * 100}%` }}>
                      <div className="ring-inner"><strong>{formatPercent(metrics?.accuracy)}</strong><span>Accuracy</span></div>
                    </div>
                    <div className="performance-bars">
                      {[
                        { label: 'Precision', value: metrics?.precision_cg, color: 'purple-fill' },
                        { label: 'Recall', value: metrics?.recall_cg, color: 'cyan-fill' },
                        { label: 'F1-score', value: metrics?.f1_cg, color: 'blue-fill' },
                      ].map((bar) => (
                        <div className="perf-bar-row" key={bar.label}>
                          <div className="perf-bar-label"><span>{bar.label}</span><strong>{formatPercent(bar.value)}</strong></div>
                          <div className="perf-track"><div className={`perf-fill ${bar.color}`} style={{ width: `${Math.max(0, Math.min(100, (bar.value || 0) * 100))}%` }} /></div>
                        </div>
                      ))}
                    </div>
                  </div>
                  <div className="panel-footnote"><CheckCircle2 size={15} /><span>Measured on held-out labeled data. Not a guarantee of real-world performance.</span></div>
                </article>

                <article className="panel model-panel">
                  <div className="panel-heading">
                    <div><div className="eyebrow">ACTIVE CLASSIFIER</div><h3>Model overview</h3><p>Production-style local inference</p></div>
                    <div className="panel-icon purple-icon"><Cpu size={19} /></div>
                  </div>
                  <div className="model-orbit"><div className="orbit-ring orbit-one" /><div className="orbit-ring orbit-two" /><div className="orbit-core"><Network size={28} /></div><span className="orbit-node node-a" /><span className="orbit-node node-b" /><span className="orbit-node node-c" /></div>
                  <div className="model-title">TF-IDF + Logistic Regression</div>
                  <p className="model-copy">A lightweight NLP pipeline that learns text patterns from labeled reviews.</p>
                  <div className="model-tags"><span><Check size={12} /> NLP pipeline</span><span><Check size={12} /> Binary classes</span><span><Check size={12} /> Saved model</span></div>
                </article>
              </section>

              <section className="bottom-grid">
                <article className="panel quick-panel">
                  <div className="panel-heading"><div><div className="eyebrow">QUICK ACTIONS</div><h3>What would you like to do?</h3><p>Jump into an analysis workflow.</p></div></div>
                  <div className="quick-actions">
                    <button className="quick-action" onClick={() => goTo('Analyze review')}><span className="quick-action-icon cyan-icon"><MessageSquareText size={19} /></span><span><strong>Analyze a review</strong><small>Run a single text prediction</small></span><ArrowRight size={16} /></button>
                    <button className="quick-action" onClick={() => goTo('Batch analysis')}><span className="quick-action-icon purple-icon"><FileSpreadsheet size={19} /></span><span><strong>Process a CSV</strong><small>Classify multiple review rows</small></span><ArrowRight size={16} /></button>
                    <button className="quick-action" onClick={() => goTo('Model performance')}><span className="quick-action-icon blue-icon"><BarChart3 size={19} /></span><span><strong>View model metrics</strong><small>Inspect evaluation and limitations</small></span><ArrowRight size={16} /></button>
                  </div>
                </article>
                <article className="panel classes-panel">
                  <div className="panel-heading"><div><div className="eyebrow">LABEL DEFINITIONS</div><h3>What the model predicts</h3><p>Classes from your training dataset.</p></div><div className="panel-icon blue-icon"><Database size={19} /></div></div>
                  <div className="class-definition"><span className="class-pill cg-pill">CG</span><div><strong>Computer-generated</strong><p>Text classified into the CG dataset label.</p></div></div>
                  <div className="class-definition"><span className="class-pill or-pill">OR</span><div><strong>Original</strong><p>Text classified into the OR dataset label.</p></div></div>
                  <div className="note-box"><ShieldAlert size={16} /><span>These classes do not prove a review is fraudulent or authentic.</span></div>
                </article>
              </section>
            </>
          )}

          {page === 'Analyze review' && (
            <>
              <SectionHeading eyebrow="SINGLE REVIEW INTELLIGENCE" title="Analyze a review" description="Submit review text and inspect the model's predicted class." action={<span className="pill-status"><span className={`status-dot ${apiStatus}`} /> {apiStatus === 'online' ? 'Model connected' : 'API unavailable'}</span>} />
              <div className="analysis-layout">
                <section className="panel input-panel">
                  <div className="panel-heading"><div><h3>Review input</h3><p>Paste a review or start with an example.</p></div><div className="panel-icon cyan-icon"><MessageSquareText size={18} /></div></div>
                  <form onSubmit={analyzeReview}>
                    <div className="field-header"><label htmlFor="review-input">Review text</label><span>{reviewText.length.toLocaleString()} / 10,000</span></div>
                    <textarea id="review-input" maxLength={10000} value={reviewText} onChange={(event) => setReviewText(event.target.value)} placeholder="Paste the customer review here..." className="review-textarea" />
                    <div className="sample-heading">Try a sample <span>Click to load</span></div>
                    <div className="sample-list">{sampleReviews.map((sample) => <button className="sample-chip" type="button" key={sample.label} onClick={() => { setReviewText(sample.text); setReviewResult(null); setPredictError('') }}>{sample.label}<ArrowDownRight size={13} /></button>)}</div>
                    {predictError && <div className="error-message"><AlertCircle size={16} />{predictError}</div>}
                    <button className="primary-button full-button" disabled={predicting || apiStatus !== 'online'} type="submit">{predicting ? <LoaderCircle className="spin" size={17} /> : <WandSparkles size={17} />}{predicting ? 'Analyzing review...' : 'Analyze review'}<ArrowRight size={15} /></button>
                  </form>
                  <div className="privacy-hint"><Shield size={15} /><span>Review text is sent to your local FastAPI backend at <code>127.0.0.1:8000</code>.</span></div>
                </section>

                <section className="panel result-panel">
                  <div className="panel-heading"><div><h3>Analysis result</h3><p>Prediction and class score</p></div><div className="panel-icon purple-icon"><Fingerprint size={18} /></div></div>
                  {!reviewResult ? (
                    <div className="empty-result"><div className="empty-orb"><ShieldCheck size={31} /></div><strong>Your result will appear here</strong><p>Enter review text on the left and run an analysis to see the predicted class.</p><div className="empty-steps"><span><Check size={13} /> Text input</span><ArrowRight size={13} /><span><Cpu size={13} /> ML inference</span><ArrowRight size={13} /><span><Activity size={13} /> Result</span></div></div>
                  ) : (
                    <div className="result-content">
                      <div className={`prediction-banner ${reviewResult.label_code === 'CG' ? 'prediction-cg' : 'prediction-or'}`}>
                        <div className="prediction-symbol">{reviewResult.label_code === 'CG' ? <ShieldAlert size={23} /> : <ShieldCheck size={23} />}</div>
                        <div><span>MODEL PREDICTION</span><strong>{reviewResult.prediction}</strong><small>Predicted label: {reviewResult.label_code}</small></div>
                      </div>
                      <div className="score-head"><span>Predicted-class score</span><strong>{Number(reviewResult.predicted_class_score).toFixed(2)}%</strong></div>
                      <div className="score-track"><div className={`score-fill ${reviewResult.label_code === 'CG' ? 'cg-fill' : 'or-fill'}`} style={{ width: `${reviewResult.predicted_class_score}%` }} /></div>
                      <p className="score-caveat">This is the model's class score, not a verified probability of fraud or authenticity.</p>
                      <div className="probability-title">Class score breakdown</div>
                      <div className="class-score-row"><div className="class-score-label"><span className="mini-label cg-pill">CG</span><span>Computer-generated</span><strong>{Number(reviewResult.cg_score).toFixed(2)}%</strong></div><div className="score-track thin"><div className="score-fill cg-fill" style={{ width: `${reviewResult.cg_score}%` }} /></div></div>
                      <div className="class-score-row"><div className="class-score-label"><span className="mini-label or-pill">OR</span><span>Original</span><strong>{Number(reviewResult.or_score).toFixed(2)}%</strong></div><div className="score-track thin"><div className="score-fill or-fill" style={{ width: `${reviewResult.or_score}%` }} /></div></div>
                      <div className="result-explanation"><CircleHelp size={16} /><span>{reviewResult.note}</span></div>
                    </div>
                  )}
                </section>
              </div>
            </>
          )}

          {page === 'Batch analysis' && (
            <>
              <SectionHeading eyebrow="BULK REVIEW PROCESSING" title="Batch CSV analysis" description="Upload a CSV, select the review-text column, and download predictions for each row." action={<span className="pill-status"><FileSpreadsheet size={14} /> Up to 50,000 rows</span>} />
              <div className="batch-layout">
                <section className="panel upload-panel">
                  <div className="panel-heading"><div><h3>Upload review dataset</h3><p>CSV format · Maximum file size 25 MB</p></div><div className="panel-icon purple-icon"><CloudUpload size={19} /></div></div>
                  <form onSubmit={runBatch}>
                    <label className={`drop-zone ${dragActive ? 'drop-zone-active' : ''}`} onDragOver={(event) => { event.preventDefault(); setDragActive(true) }} onDragLeave={() => setDragActive(false)} onDrop={(event) => { event.preventDefault(); setDragActive(false); const file = event.dataTransfer.files?.[0]; if (file) setBatchFile(file) }}>
                      <input type="file" accept=".csv,text/csv" onChange={(event) => setBatchFile(event.target.files?.[0] || null)} hidden />
                      <div className="upload-orb"><Upload size={23} /></div>
                      <strong>{batchFile ? batchFile.name : 'Drop your CSV file here'}</strong>
                      <span>{batchFile ? `${(batchFile.size / 1024).toFixed(1)} KB · Ready to analyze` : 'or click to browse from your computer'}</span>
                      <small>CSV only · Keep column headers in the first row</small>
                    </label>
                    <div className="form-field"><label htmlFor="text-column">Review-text column</label><select id="text-column" value={textColumn} onChange={(event) => setTextColumn(event.target.value)}><option value="auto">Auto-detect (text_ / Review_Text)</option><option value="text_">text_</option><option value="Review_Text">Review_Text</option><option value="review_text">review_text</option><option value="review">review</option><option value="text">text</option></select><span className="field-help">Choose the exact column name from your CSV.</span></div>
                    {batchError && <div className="error-message"><AlertCircle size={16} />{batchError}</div>}
                    <button className="primary-button full-button" disabled={batching || apiStatus !== 'online'} type="submit">{batching ? <LoaderCircle className="spin" size={17} /> : <Zap size={17} />}{batching ? 'Processing reviews...' : 'Run batch analysis'}<ArrowRight size={15} /></button>
                  </form>
                </section>
                <section className="panel batch-info-panel">
                  <div className="panel-heading"><div><h3>How batch analysis works</h3><p>From a raw CSV to an enriched result file.</p></div><div className="panel-icon cyan-icon"><Network size={18} /></div></div>
                  <div className="pipeline-steps">
                    <div className="pipeline-step"><div className="pipeline-number">01</div><div><strong>Upload data</strong><p>Select a CSV containing review text.</p></div></div>
                    <div className="pipeline-line" />
                    <div className="pipeline-step"><div className="pipeline-number">02</div><div><strong>Run model inference</strong><p>TF-IDF and Logistic Regression classify each non-empty row.</p></div></div>
                    <div className="pipeline-line" />
                    <div className="pipeline-step"><div className="pipeline-number">03</div><div><strong>Export predictions</strong><p>Download the original rows plus prediction and score columns.</p></div></div>
                  </div>
                  <div className="note-box"><ShieldAlert size={16} /><span>Batch predictions are dataset-label classifications, not verified fraud findings.</span></div>
                </section>
              </div>
              {batchResult && (
                <section className="panel batch-results-panel">
                  <div className="panel-heading"><div><div className="eyebrow">ANALYSIS COMPLETE</div><h3>{batchResult.filename}</h3><p>Text column: <code>{batchResult.text_column}</code></p></div><button className="primary-button" onClick={downloadBatch}><Download size={16} /> Download results</button></div>
                  <div className="batch-summary-grid"><div className="summary-tile"><span>Rows in file</span><strong>{formatNumber(batchResult.total_rows)}</strong></div><div className="summary-tile"><span>Analyzed</span><strong>{formatNumber(batchResult.analyzed_rows)}</strong></div><div className="summary-tile cg-summary"><span>CG predictions</span><strong>{formatNumber(batchResult.cg_count)}</strong></div><div className="summary-tile or-summary"><span>OR predictions</span><strong>{formatNumber(batchResult.or_count)}</strong></div></div>
                  <div className="table-wrap"><table><thead><tr>{Object.keys(batchResult.rows?.[0] || {}).slice(0, 7).map((key) => <th key={key}>{key.replaceAll('_', ' ')}</th>)}</tr></thead><tbody>{batchResult.rows.slice(0, 8).map((row, index) => <tr key={index}>{Object.keys(batchResult.rows[0]).slice(0, 7).map((key) => <td key={key}>{String(row[key] ?? '').slice(0, 90)}</td>)}</tr>)}</tbody></table></div>
                  <div className="panel-footnote"><CheckCircle2 size={15} /><span>{batchResult.note}</span></div>
                </section>
              )}
            </>
          )}

          {page === 'Model performance' && (
            <>
              <SectionHeading eyebrow="MODEL EVALUATION" title="Model performance" description="A transparent view of how the current classifier performed on held-out labeled examples." action={<span className="pill-status"><Activity size={14} /> Test split</span>} />
              <section className="metric-grid performance-metrics">{metricCards.map((card) => <MetricCard key={card.label} {...card} />)}</section>
              <div className="overview-grid model-insights-grid">
                <article className="panel performance-panel">
                  <div className="panel-heading"><div><div className="eyebrow">METRIC BREAKDOWN</div><h3>Classification quality</h3><p>Positive class: computer-generated (CG)</p></div><div className="panel-icon cyan-icon"><BarChart3 size={18} /></div></div>
                  <div className="large-bars">{[
                    { label: 'Accuracy', value: metrics?.accuracy, color: 'cyan-fill', description: 'Correct predictions across both classes' },
                    { label: 'Precision · CG', value: metrics?.precision_cg, color: 'purple-fill', description: 'Share of predicted CG examples that were CG in the test labels' },
                    { label: 'Recall · CG', value: metrics?.recall_cg, color: 'blue-fill', description: 'Share of test CG examples the model identified' },
                    { label: 'F1-score · CG', value: metrics?.f1_cg, color: 'green-fill', description: 'Harmonic mean of CG precision and recall' },
                  ].map((item) => <div className="large-bar-item" key={item.label}><div className="large-bar-header"><strong>{item.label}</strong><span>{formatPercent(item.value)}</span></div><div className="perf-track large-track"><div className={`perf-fill ${item.color}`} style={{ width: `${Math.max(0, Math.min(100, (item.value || 0) * 100))}%` }} /></div><p>{item.description}</p></div>)}</div>
                </article>
                <article className="panel matrix-panel">
                  <div className="panel-heading"><div><div className="eyebrow">ERROR ANALYSIS</div><h3>Confusion matrix</h3><p>Counts on the held-out test set.</p></div><div className="panel-icon purple-icon"><Microscope size={18} /></div></div>
                  {metrics?.confusion_matrix_available ? <img className="confusion-image" src={`${API_BASE}/api/confusion-matrix`} alt="Confusion matrix of original and computer-generated review classes" /> : <div className="matrix-placeholder"><BarChart3 size={28} /><p>Confusion matrix image unavailable.</p><span>Run python src/train.py to regenerate it.</span></div>}
                </article>
              </div>
              <section className="panel limitation-panel"><div className="limitation-heading"><ShieldAlert size={19} /><div><h3>Interpret results responsibly</h3><p>The model predicts the labels available in its training dataset.</p></div></div><div className="limitation-grid"><div><strong>CG ≠ automatically fraudulent</strong><p>A computer-generated review is not necessarily a deceptive review.</p></div><div><strong>OR ≠ verified authentic</strong><p>An original label does not prove the content is true or unbiased.</p></div><div><strong>Test score ≠ deployment guarantee</strong><p>Performance may change on other platforms, products, or writing styles.</p></div></div></section>
            </>
          )}

          <footer className="page-footer"><div><span className="footer-mark"><ShieldCheck size={15} /></span><strong>ReviewGuard</strong><span className="footer-separator">·</span><span>NLP & Machine Learning</span></div><div><span className={`status-dot ${apiStatus}`} /> {apiStatus === 'online' ? 'API connected' : apiStatus === 'checking' ? 'Connecting…' : 'Waiting for API'} <span className="footer-separator">·</span> v1.0.0</div></footer>
        </div>
      </main>
    </div>
  )
}

export default App
