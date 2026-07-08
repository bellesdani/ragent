/* RAGent · Panel de administración */
'use strict';

/* ============ Constantes ============ */

const VIEWS = {
  overview: { section: 'Operación', title: 'Vista general' },
  agents: { section: 'Operación', title: 'Agentes' },
  sources: { section: 'Conocimiento', title: 'Fuentes de conocimiento' },
  ingest: { section: 'Conocimiento', title: 'Ingesta' },
  search: { section: 'Conocimiento', title: 'Búsqueda directa' },
};

// Fuentes que aceptan ingesta desde JSON (manuals solo acepta from-html)
const JSON_INGEST_SOURCES = ['devices', 'employees', 'tickets', 'articles'];

// Ejemplos de ingesta por fuente: cambian al seleccionar la fuente.
// Cada ejemplo es válido contra el modelo Pydantic que valida esa ingesta.
const INGEST_EXAMPLES = {
  devices: `[
  {
    "id": 1284,
    "name": "PC-Produccion-04",
    "type": "Ordenador",
    "hostname": "EQ-PROD-04",
    "operating_system": "Windows 11 Pro",
    "architecture": "x64",
    "manufacturer": "Dell",
    "model": "OptiPlex 7010",
    "serial_number": "5XK9Q34",
    "owner": "María Ferrer Ibáñez",
    "user": "maria.ferrer",
    "location": "Planta de producción",
    "ip_addresses": ["192.168.10.44"],
    "mac_addresses": ["A4:BB:6D:2E:91:0C"],
    "vlans": ["10"],
    "comments": "Equipo junto a la línea 2"
  }
]`,
  employees: `[
  {
    "id": 342,
    "first_name": "María",
    "last_name": "Ferrer Ibáñez",
    "full_name": "María Ferrer Ibáñez",
    "alias": "mferrer",
    "department": "Informática",
    "emails": ["maria.ferrer@empresa.com"],
    "phones": [
      { "number": "964000000", "extension": "213" }
    ]
  }
]`,
  tickets: `[
  {
    "ticket_id": 48213,
    "ticket_group_id": 1,
    "ticket_priority_id": 2,
    "ticket_state_id": 4,
    "ticket_number": "20260502-000123",
    "ticket_title": "Pantalla negra al arrancar",
    "ticket_created_at": "2026-05-02T08:14:00",
    "ticket_closed_at": "2026-05-02T15:40:00",
    "ticket_customer_firstname": "Laura",
    "ticket_customer_lastname": "Gómez",
    "ticket_customer_department": "Producción",
    "ticket_customer_email": "laura.gomez@empresa.com",
    "article_id": 91201,
    "article_from": "laura.gomez@empresa.com",
    "article_to": "helpdesk@empresa.com",
    "article_subject": "Pantalla negra al arrancar",
    "article_content_type": "text/html",
    "article_body": "Buenos días, al encender el portátil se queda la pantalla en negro tras la actualización de ayer.",
    "article_internal": false,
    "article_created_at": "2026-05-02T08:14:00"
  }
]`,
  articles: `[
  {
    "id": "VEN028415",
    "descripcion": "Azulejo Artisan White 13,2x13,2",
    "unidad_principal": "M2",
    "usuario_alta": "mferrer",
    "en_catalogo": -1,
    "fecha_alta": "2024-03-12T09:30:00",
    "adn": "ARTISAN",
    "referencia_venta": "ART-WH-1313",
    "ean13": "8435123456789",
    "familia_id": "AZU",
    "familia_descripcion": "Azulejos",
    "subfamilia_id": "AZB",
    "subfamilia_descripcion": "Azulejo brillo",
    "formato_id": 13,
    "formato_descripcion": "13,2x13,2"
  }
]`,
};

const INGEST_HINTS = {
  devices: 'Lista JSON de dispositivos: equipos, impresoras y servidores. Cada objeto necesita al menos su "id" numérico.',
  employees: 'Lista JSON de empleados con su contacto corporativo: emails, teléfonos y extensiones.',
  tickets: 'Lista JSON de filas ticket-artículo de HelpDesk. Las filas del mismo "ticket_id" se agrupan en un único punto y se resumen con el agente Summarizer.',
  articles: 'Lista JSON de artículos del ERP (PL_ARTICULOS) con sus claves originales en castellano: "descripcion", "familia_id", etc.',
};

const ACTIVITY_COLORS = { ingest: '#4b57d6', html: '#0b8f6f', search: '#c58b2f', create: '#8c92a8' };
const ACTIVITY_KEY = 'ragent_activity';

/* ============ Estado ============ */

const state = {
  view: 'overview',
  overview: null,
  agents: [],
  sources: [],
  selectedAgent: null,
  selectedSource: null,
  ingestSource: 'devices',
  ingestEdited: false,
  htmlFile: null,
  searchSource: 'tickets',
  searchLimit: 5,
  searchResponse: null,
  selectedResult: 0,
};

const $ = (id) => document.getElementById(id);

/* ============ Utilidades ============ */

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;');
}

// Convierte un valor JSON en HTML resaltado (claves, cadenas y números)
function highlightJson(value) {
  const json = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  return escapeHtml(json).replace(
    /("(?:\\.|[^"\\])*")(\s*:)?|(-?\b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b)|\b(true|false|null)\b/g,
    (match, str, colon, num, kw) => {
      if (str) return colon ? `<span class="k">${str}</span>${colon}` : `<span class="s">${str}</span>`;
      if (num) return `<span class="n">${num}</span>`;
      return `<span class="n">${kw}</span>`;
    },
  );
}

function fmtInt(value) {
  if (value === null || value === undefined) return '—';
  return Number(value).toLocaleString('es-ES');
}

function fmtBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function relativeTime(timestamp) {
  const diff = Date.now() - timestamp;
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return 'ahora mismo';
  if (minutes < 60) return `hace ${minutes} min`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `hace ${hours} h`;
  const days = Math.floor(hours / 24);
  if (days === 1) return 'ayer';
  return `hace ${days} días`;
}

function toast(message, kind = '') {
  const el = document.createElement('div');
  el.className = `toast ${kind}`;
  el.textContent = message;
  $('toasts').appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let data = null;
  try { data = await response.json(); } catch { /* respuesta sin cuerpo JSON */ }
  if (!response.ok) {
    const message = data?.error?.message || data?.detail || `Error ${response.status}`;
    throw new Error(message);
  }
  return data;
}

/* ============ Actividad reciente (registro local del panel) ============ */

function loadActivity() {
  try { return JSON.parse(localStorage.getItem(ACTIVITY_KEY)) || []; } catch { return []; }
}

function logActivity(type, html, meta) {
  const items = loadActivity();
  items.unshift({ type, html, meta, ts: Date.now() });
  localStorage.setItem(ACTIVITY_KEY, JSON.stringify(items.slice(0, 20)));
  renderActivity();
}

function renderActivity() {
  const items = loadActivity();
  const container = $('activity-list');
  if (!items.length) {
    container.innerHTML = '<div class="activity-empty">Sin actividad todavía. Las ingestas, búsquedas y colecciones creadas desde este panel aparecerán aquí.</div>';
    return;
  }
  container.innerHTML = items.slice(0, 8).map((item) => `
    <div class="activity-item">
      <span class="activity-dot" style="background:${ACTIVITY_COLORS[item.type] || '#8c92a8'}"></span>
      <div>
        <div class="activity-text">${item.html}</div>
        <div class="activity-time">${relativeTime(item.ts)}${item.meta ? ` · ${escapeHtml(item.meta)}` : ''}</div>
      </div>
    </div>
  `).join('');
}

/* ============ Router ============ */

function currentViewFromHash() {
  const key = (location.hash || '#/overview').replace('#/', '').split('?')[0];
  return VIEWS[key] ? key : 'overview';
}

function navigate() {
  state.view = currentViewFromHash();
  document.querySelectorAll('.nav-item').forEach((item) => {
    item.classList.toggle('active', item.dataset.view === state.view);
  });
  document.querySelectorAll('.view').forEach((view) => {
    view.hidden = view.id !== `view-${state.view}`;
  });
  $('crumb-section').textContent = VIEWS[state.view].section;
  $('crumb-title').textContent = VIEWS[state.view].title;
}

/* ============ Carga de datos ============ */

async function loadHealth() {
  const chip = $('health-chip');
  try {
    await api('/health');
    chip.textContent = 'GET /health · 200';
    chip.classList.remove('down');
  } catch {
    chip.textContent = 'GET /health · sin respuesta';
    chip.classList.add('down');
  }
}

async function loadOverview() {
  try {
    state.overview = await api('/admin/overview');
  } catch (error) {
    toast(`No se pudo cargar la vista general: ${error.message}`, 'error');
    state.overview = null;
  }
  renderOverview();
  renderSidebarStatus();
}

async function loadAgents() {
  try {
    const data = await api('/admin/agents');
    state.agents = data.items;
    if (!state.selectedAgent && state.agents.length) state.selectedAgent = state.agents[0].agent_id;
  } catch (error) {
    toast(`No se pudieron cargar los agentes: ${error.message}`, 'error');
  }
  renderAgents();
}

async function loadSources() {
  try {
    const data = await api('/knowledge-source');
    state.sources = data.result.items;
    if (!state.selectedSource && state.sources.length) state.selectedSource = state.sources[0].id;
  } catch (error) {
    toast(`No se pudieron cargar las fuentes: ${error.message}`, 'error');
  }
  renderSources();
  renderSearchChips();
  renderSearchRetrievalBadge();
}

/* ============ Vista general ============ */

function renderSidebarStatus() {
  const overview = state.overview;
  const dot = $('qdrant-dot');
  if (!overview) {
    dot.className = 'status-dot down';
    $('qdrant-status-label').textContent = 'Qdrant sin datos';
    return;
  }
  const ok = overview.qdrant.status === 'ok';
  dot.className = `status-dot ${ok ? 'ok' : 'down'}`;
  $('qdrant-status-label').textContent = ok ? 'Qdrant operativo' : 'Qdrant no disponible';
  $('qdrant-url-label').textContent = overview.qdrant.url.replace(/^https?:\/\//, '');
  $('version-label').textContent = `${overview.service.name.toLowerCase()} · v${overview.service.version}`;
}

function statCard(label, value, sub, indicator) {
  return `
    <div class="stat-card">
      <div class="stat-card-top">
        <span class="stat-card-label">${escapeHtml(label)}</span>
        ${indicator}
      </div>
      <div class="stat-card-value">${escapeHtml(value)}</div>
      <div class="stat-card-sub" title="${escapeHtml(sub)}">${escapeHtml(sub)}</div>
    </div>`;
}

function renderOverview() {
  const overview = state.overview;
  const cards = $('overview-cards');
  if (!overview) {
    cards.innerHTML = statCard('Servicio', 'Sin conexión', 'No se pudo consultar /admin/overview', '<span class="status-dot down"></span>');
    $('overview-sources-table').innerHTML = '';
    $('overview-sources-note').textContent = '';
    renderActivity();
    return;
  }

  const qdrantOk = overview.qdrant.status === 'ok';
  cards.innerHTML = [
    statCard('Servicio', 'Operativo', `${overview.service.name} v${overview.service.version}`, '<span class="status-dot ok"></span>'),
    statCard(
      'Qdrant',
      qdrantOk ? `${overview.qdrant.collections_total} colecciones` : 'Sin conexión',
      overview.qdrant.url.replace(/^https?:\/\//, ''),
      `<span class="status-dot ${qdrantOk ? 'ok' : 'down'}"></span>`,
    ),
    statCard(
      'Agentes',
      `${overview.agents.total} definidos`,
      overview.agents.names.join(' · '),
      `<span class="badge badge-primary">${overview.agents.public} público${overview.agents.public === 1 ? '' : 's'}</span>`,
    ),
    statCard(
      'Backend de chat',
      overview.chat_backend.model,
      `temp ${overview.chat_backend.temperature} · max ${overview.chat_backend.max_tokens} tok`,
      '<span class="status-dot ok"></span>',
    ),
  ].join('');

  $('overview-sources-note').textContent = `${overview.sources.length} fuentes · recuperación híbrida`;
  const rows = overview.sources.map((source) => `
    <div class="src-row">
      <div class="src-col-name">
        <div class="src-name">${escapeHtml(source.name)}</div>
        <div class="src-desc" title="${escapeHtml(source.description)}">${escapeHtml(source.description)}</div>
      </div>
      <div class="src-col-collection">${escapeHtml(source.collection_name)}</div>
      <div class="src-col-retrieval"><span class="badge badge-retrieval">${escapeHtml(source.retrieval_type)}</span></div>
      <div class="src-col-points">${source.collection_exists ? fmtInt(source.points) : '—'}</div>
      <div class="src-col-updated">${source.collection_exists ? escapeHtml(source.last_update || '—') : 'sin colección'}</div>
    </div>
  `).join('');
  $('overview-sources-table').innerHTML = `
    <div class="src-row-head">
      <div class="src-col-name">Fuente</div>
      <div class="src-col-collection">Colección</div>
      <div class="src-col-retrieval">Retrieval</div>
      <div class="src-col-points">Puntos</div>
      <div class="src-col-updated">Actualizado</div>
    </div>${rows}`;

  renderActivity();
}

/* ============ Agentes ============ */

function agentBadge(agent) {
  if (agent.public) return '<span class="badge badge-public">Público</span>';
  if (agent.agent_id === 'Summarizer') return '<span class="badge badge-aux">Auxiliar</span>';
  return '<span class="badge badge-internal">Interno</span>';
}

function renderAgents() {
  const list = $('agents-list');
  list.innerHTML = state.agents.map((agent) => `
    <div class="list-card ${agent.agent_id === state.selectedAgent ? 'selected' : ''}" data-agent="${escapeHtml(agent.agent_id)}">
      <div class="list-card-top">
        <div class="list-card-title">${escapeHtml(agent.name)}</div>
        ${agentBadge(agent)}
      </div>
      <div class="list-card-sub">${escapeHtml(agent.description)}</div>
    </div>
  `).join('');
  list.querySelectorAll('.list-card').forEach((card) => {
    card.addEventListener('click', () => {
      state.selectedAgent = card.dataset.agent;
      renderAgents();
    });
  });

  const agent = state.agents.find((item) => item.agent_id === state.selectedAgent);
  const detail = $('agent-detail');
  if (!agent) {
    detail.innerHTML = '<p class="detail-desc">No hay agentes definidos en el catálogo.</p>';
    return;
  }

  const toolChips = [
    ...agent.tools.retrieval.map((name) => `<span class="tool-chip">${escapeHtml(name)}</span>`),
    ...agent.tools.utility.map((name) => `<span class="tool-chip utility">${escapeHtml(name)}</span>`),
  ].join('');
  const totalTools = agent.tools.retrieval.length + agent.tools.utility.length;
  const toolsSection = agent.enable_tools
    ? `
      <div class="section-label">Herramientas registradas · ${totalTools}</div>
      <div class="tool-chips">${toolChips}</div>
      <div class="tool-chips-note">${agent.tools.retrieval.length} herramientas de retrieval sobre fuentes de conocimiento · ${agent.tools.utility.length} utilidades</div>`
    : `
      <div class="section-label">Herramientas registradas · 0</div>
      <div class="tool-chips-note">Este agente no registra herramientas: responde únicamente con el modelo base y su prompt.</div>`;

  detail.innerHTML = `
    <div class="detail-header">
      <div>
        <div class="detail-title-row">
          <h2>${escapeHtml(agent.name)}</h2>
          ${agentBadge(agent)}
          ${agent.enable_tools ? '<span class="badge badge-primary">Herramientas activas</span>' : ''}
        </div>
        <p class="detail-desc">${escapeHtml(agent.description)}</p>
      </div>
    </div>
    <div class="prop-grid">
      <div class="prop-cell"><div class="prop-label">agent_id</div><div class="prop-value">${escapeHtml(agent.agent_id)}</div></div>
      <div class="prop-cell"><div class="prop-label">modelo</div><div class="prop-value">${escapeHtml(agent.model)}</div></div>
      <div class="prop-cell"><div class="prop-label">public</div><div class="prop-value ${agent.public}">${agent.public}</div></div>
      <div class="prop-cell"><div class="prop-label">enable_tools</div><div class="prop-value ${agent.enable_tools}">${agent.enable_tools}</div></div>
    </div>
    <div class="section-label">System prompt ${agent.system_prompt_file ? `<span class="mono">${escapeHtml(agent.system_prompt_file)}</span>` : ''}</div>
    <div class="codeblock codeblock-gap">${escapeHtml(agent.system_prompt)}</div>
    ${toolsSection}`;
}

/* ============ Fuentes ============ */

function sourceStatus(sourceId) {
  return state.overview?.sources.find((item) => item.id === sourceId) || null;
}

function renderSources() {
  const list = $('sources-list');
  list.innerHTML = state.sources.map((source) => {
    const status = sourceStatus(source.id);
    const points = status?.collection_exists ? `${fmtInt(status.points)} puntos` : 'sin colección';
    return `
      <div class="list-card ${source.id === state.selectedSource ? 'selected' : ''}" data-source="${escapeHtml(source.id)}">
        <div class="list-card-top">
          <div class="list-card-title">${escapeHtml(source.name)}</div>
          <span class="badge badge-retrieval">${escapeHtml(source.retrieval_type)}</span>
        </div>
        <div class="list-card-sub mono">${points}</div>
      </div>`;
  }).join('');
  list.querySelectorAll('.list-card').forEach((card) => {
    card.addEventListener('click', () => {
      state.selectedSource = card.dataset.source;
      renderSources();
    });
  });

  const source = state.sources.find((item) => item.id === state.selectedSource);
  const detail = $('source-detail');
  if (!source) {
    detail.innerHTML = '<p class="detail-desc">No hay fuentes de conocimiento configuradas.</p>';
    return;
  }

  const status = sourceStatus(source.id);
  const statusBadge = status
    ? (status.collection_exists
      ? '<span class="badge badge-internal">Colección creada</span>'
      : '<span class="badge badge-aux">Colección no creada</span>')
    : '';

  detail.innerHTML = `
    <div class="detail-header">
      <div>
        <div class="detail-title-row">
          <h2>${escapeHtml(source.name)}</h2>
          <span class="badge badge-retrieval">${escapeHtml(source.retrieval_type)}</span>
          ${statusBadge}
        </div>
        <p class="detail-desc">${escapeHtml(source.description)}</p>
      </div>
      <div class="detail-actions">
        <button class="btn btn-ghost" id="source-go-search">Buscar</button>
        <button class="btn btn-primary" id="source-create">Crear colección</button>
      </div>
    </div>
    <div class="prop-grid prop-grid-2">
      <div class="prop-cell"><div class="prop-label">collection_name</div><div class="prop-value">${escapeHtml(source.collection_name)}</div></div>
      <div class="prop-cell"><div class="prop-label">retrieval_type</div><div class="prop-value">${escapeHtml(source.retrieval_type)}</div></div>
      <div class="prop-cell"><div class="prop-label">dense_vector_name</div><div class="prop-value">${escapeHtml(source.dense_vector_name ?? '—')}</div></div>
      <div class="prop-cell"><div class="prop-label">sparse_vector_name</div><div class="prop-value">${escapeHtml(source.sparse_vector_name ?? '—')}</div></div>
    </div>
    <div class="section-label">payload_keys</div>
    <div class="codeblock">${highlightJson(source.payload_keys)}</div>`;

  $('source-go-search').addEventListener('click', () => {
    state.searchSource = source.id;
    renderSearchChips();
    renderSearchRetrievalBadge();
    location.hash = '#/search';
    $('search-query').focus();
  });
  $('source-create').addEventListener('click', async () => {
    const button = $('source-create');
    button.disabled = true;
    try {
      const data = await api(`/knowledge-source/${source.id}`, { method: 'POST' });
      const created = data.result.collection_created;
      toast(created ? `Colección "${source.collection_name}" creada` : `La colección "${source.collection_name}" ya existía`, 'success');
      if (created) {
        logActivity('create', `Colección creada · <span class="mono">${escapeHtml(source.collection_name)}</span>`);
      }
      await loadOverview();
      renderSources();
    } catch (error) {
      toast(`Error creando la colección: ${error.message}`, 'error');
    } finally {
      button.disabled = false;
    }
  });
}

/* ============ Ingesta ============ */

function isPristineIngestEditor() {
  const value = $('ingest-json-editor').value.trim();
  return value === '' || Object.values(INGEST_EXAMPLES).some((example) => example.trim() === value);
}

function renderIngestChips() {
  $('ingest-source-chips').innerHTML = JSON_INGEST_SOURCES.map((id) => `
    <button class="select-chip ${id === state.ingestSource ? 'selected' : ''}" data-source="${id}">${id}</button>
  `).join('');
  $('ingest-source-chips').querySelectorAll('.select-chip').forEach((chip) => {
    chip.addEventListener('click', () => selectIngestSource(chip.dataset.source));
  });
  $('ingest-json-hint').textContent = INGEST_HINTS[state.ingestSource];
}

function selectIngestSource(sourceId) {
  state.ingestSource = sourceId;
  // El ejemplo cambia según la fuente elegida; el contenido escrito a mano se respeta
  if (isPristineIngestEditor()) {
    $('ingest-json-editor').value = INGEST_EXAMPLES[sourceId];
    state.ingestEdited = false;
    validateIngestJson();
  }
  renderIngestChips();
}

function validateIngestJson() {
  const statusEl = $('ingest-json-status');
  const value = $('ingest-json-editor').value.trim();
  if (!value) {
    statusEl.textContent = 'Pega una lista JSON o carga el ejemplo';
    statusEl.classList.remove('error');
    return null;
  }
  try {
    const parsed = JSON.parse(value);
    if (!Array.isArray(parsed)) {
      statusEl.textContent = 'El JSON debe ser una lista de objetos';
      statusEl.classList.add('error');
      return null;
    }
    statusEl.textContent = `${parsed.length} objeto${parsed.length === 1 ? '' : 's'} · listo para indexar`;
    statusEl.classList.remove('error');
    return parsed;
  } catch {
    statusEl.textContent = 'JSON no válido';
    statusEl.classList.add('error');
    return null;
  }
}

function showIngestResult(sourceId, data) {
  $('ingest-result-card').hidden = false;
  $('ingest-result-badge').className = 'badge badge-ok';
  $('ingest-result-badge').textContent = `status: ${data.status} · ${data.operation}`;
  $('ingest-result-points').textContent = fmtInt(data.result.points);
  $('ingest-result-json').innerHTML = highlightJson({ knowledge_source_id: sourceId, ...data.result });
}

async function submitIngestJson() {
  const parsed = validateIngestJson();
  if (!parsed) {
    toast('Revisa el JSON antes de insertar los puntos', 'error');
    return;
  }
  const sourceId = state.ingestSource;
  const button = $('ingest-json-submit');
  button.disabled = true;
  button.textContent = 'Insertando…';
  try {
    const data = await api(`/knowledge-source/${sourceId}/points/from-json`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(parsed),
    });
    showIngestResult(sourceId, data);
    toast(`Ingesta completada: ${data.result.points} puntos en ${sourceId}`, 'success');
    logActivity('ingest', `<b>Ingesta</b> en <span class="mono">${escapeHtml(sourceId)}</span> · +${data.result.points} puntos`, 'from-json');
    loadOverview();
  } catch (error) {
    toast(`Error en la ingesta: ${error.message}`, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Insertar puntos';
  }
}

function setHtmlFile(file) {
  if (file && !/\.html?$/i.test(file.name) && file.type !== 'text/html') {
    toast('El fichero debe ser un HTML', 'error');
    return;
  }
  state.htmlFile = file || null;
  $('html-file-chip').hidden = !file;
  $('ingest-html-submit').disabled = !file;
  if (file) {
    $('html-file-name').textContent = file.name;
    $('html-file-meta').textContent = `${fmtBytes(file.size)} · ${file.type || 'text/html'}`;
  }
}

async function submitIngestHtml() {
  if (!state.htmlFile) return;
  const button = $('ingest-html-submit');
  button.disabled = true;
  button.textContent = 'Subiendo…';
  try {
    const form = new FormData();
    form.append('file', state.htmlFile);
    const data = await api('/knowledge-source/manuals/points/from-html', { method: 'POST', body: form });
    showIngestResult('manuals', data);
    toast(`Manual indexado: ${data.result.points} puntos en manuals`, 'success');
    logActivity('html', `Manual HTML subido · <span class="mono">${escapeHtml(state.htmlFile.name)}</span>`, `from-html · ${data.result.points} puntos`);
    setHtmlFile(null);
    loadOverview();
  } catch (error) {
    toast(`Error subiendo el manual: ${error.message}`, 'error');
  } finally {
    button.disabled = !state.htmlFile;
    button.textContent = 'Subir manual';
  }
}

function setupIngest() {
  renderIngestChips();
  $('ingest-json-editor').value = INGEST_EXAMPLES[state.ingestSource];
  validateIngestJson();

  $('ingest-json-editor').addEventListener('input', () => {
    state.ingestEdited = true;
    validateIngestJson();
  });
  $('ingest-load-example').addEventListener('click', () => {
    $('ingest-json-editor').value = INGEST_EXAMPLES[state.ingestSource];
    state.ingestEdited = false;
    validateIngestJson();
  });
  $('ingest-json-submit').addEventListener('click', submitIngestJson);

  const dropzone = $('html-dropzone');
  dropzone.addEventListener('click', () => $('html-file-input').click());
  dropzone.addEventListener('dragover', (event) => { event.preventDefault(); dropzone.classList.add('dragover'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
  dropzone.addEventListener('drop', (event) => {
    event.preventDefault();
    dropzone.classList.remove('dragover');
    setHtmlFile(event.dataTransfer.files[0]);
  });
  $('html-file-input').addEventListener('change', (event) => setHtmlFile(event.target.files[0]));
  $('html-file-remove').addEventListener('click', () => {
    setHtmlFile(null);
    $('html-file-input').value = '';
  });
  $('ingest-html-submit').addEventListener('click', submitIngestHtml);
}

/* ============ Búsqueda ============ */

function renderSearchChips() {
  $('search-source-chips').innerHTML = state.sources.map((source) => `
    <button class="select-chip ${source.id === state.searchSource ? 'selected' : ''}" data-source="${escapeHtml(source.id)}">${escapeHtml(source.id)}</button>
  `).join('');
  $('search-source-chips').querySelectorAll('.select-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      state.searchSource = chip.dataset.source;
      renderSearchChips();
      renderSearchRetrievalBadge();
    });
  });
}

function renderSearchRetrievalBadge() {
  const source = state.sources.find((item) => item.id === state.searchSource);
  $('search-retrieval-badge').textContent = source ? source.retrieval_type : 'hybrid';
}

function resultTitle(item) {
  const metadata = item.metadata || {};
  return metadata.title || metadata.full_name || metadata.name || metadata.descripcion || metadata.description || (item.content || '').split('\n')[0] || item.id;
}

function resultRef(item) {
  const metadata = item.metadata || {};
  if (metadata.number) return `#${metadata.number}`;
  if (metadata.id !== undefined) return String(metadata.id);
  return item.id;
}

async function submitSearch() {
  const query = $('search-query').value.trim();
  if (!query) {
    toast('Escribe una consulta para buscar', 'error');
    $('search-query').focus();
    return;
  }
  const button = $('search-submit');
  button.disabled = true;
  button.textContent = 'Buscando…';
  try {
    const data = await api(`/knowledge-source/${state.searchSource}/search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, limit: state.searchLimit }),
    });
    state.searchResponse = data;
    state.selectedResult = 0;
    renderSearchResults();
    logActivity('search', `Búsqueda en <span class="mono">${escapeHtml(state.searchSource)}</span> · «${escapeHtml(query)}»`, `${data.result.count} resultados`);
  } catch (error) {
    toast(`Error en la búsqueda: ${error.message}`, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Buscar';
  }
}

function renderSearchResults() {
  const response = state.searchResponse;
  const hasResults = Boolean(response);
  $('search-results').hidden = !hasResults;
  $('search-empty').hidden = hasResults;
  if (!response) return;

  const result = response.result;
  $('search-results-meta').innerHTML = `<b>${result.count} resultado${result.count === 1 ? '' : 's'}</b> · por score${result.last_data_update ? ` · <span class="mono">last_data_update ${escapeHtml(result.last_data_update)}</span>` : ''}`;

  $('search-results-list').innerHTML = result.items.map((item, index) => `
    <div class="list-card ${index === state.selectedResult ? 'selected' : ''}" data-index="${index}">
      <div class="result-card-head">
        <div class="result-card-ids">
          <span class="result-index">${String(index + 1).padStart(2, '0')}</span>
          <span class="result-ref">${escapeHtml(resultRef(item))}</span>
        </div>
        <span class="result-score">${item.score.toFixed(3)}</span>
      </div>
      <p class="result-snippet">${escapeHtml((item.content || '').slice(0, 220))}</p>
    </div>
  `).join('');
  $('search-results-list').querySelectorAll('.list-card').forEach((card) => {
    card.addEventListener('click', () => {
      state.selectedResult = Number(card.dataset.index);
      renderSearchResults();
    });
  });

  const detail = $('search-result-detail');
  const item = result.items[state.selectedResult];
  if (!item) {
    detail.innerHTML = '<p class="detail-desc">La búsqueda no ha devuelto documentos para esta consulta.</p>';
    return;
  }

  const scorePct = Math.max(0, Math.min(1, item.score)) * 100;
  detail.innerHTML = `
    <div class="detail-header">
      <div>
        <div class="detail-title-row" style="margin-bottom:8px">
          <span class="badge badge-primary">${escapeHtml(response.knowledge_source_id)}</span>
          <span class="mono" style="font-size:12.5px;color:var(--muted)">${escapeHtml(resultRef(item))}</span>
        </div>
        <h2 style="font-size:17px;line-height:1.3;letter-spacing:-.01em">${escapeHtml(resultTitle(item))}</h2>
      </div>
      <div class="score-ring-wrap">
        <div class="score-ring" style="background:conic-gradient(var(--primary) ${scorePct}%, var(--primary-bg) 0)">
          <div class="score-ring-inner">${item.score >= 1 ? item.score.toFixed(1) : `.${Math.round(item.score * 100)}`}</div>
        </div>
        <div class="score-ring-caption">SCORE</div>
      </div>
    </div>
    <div class="section-label">Contenido</div>
    <p class="result-content">${escapeHtml(item.content || '—')}</p>
    <div class="section-label">Metadata</div>
    <div class="codeblock codeblock-gap">${highlightJson(item.metadata)}</div>
    <div class="detail-footer">
      <button class="btn btn-secondary" id="result-copy-json">Copiar JSON</button>
    </div>`;

  $('result-copy-json').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(item, null, 2));
      toast('Documento copiado al portapapeles', 'success');
    } catch {
      toast('No se pudo copiar al portapapeles', 'error');
    }
  });
}

function setupSearch() {
  $('search-limit-dec').addEventListener('click', () => {
    state.searchLimit = Math.max(1, state.searchLimit - 1);
    $('search-limit').textContent = state.searchLimit;
  });
  $('search-limit-inc').addEventListener('click', () => {
    state.searchLimit = Math.min(50, state.searchLimit + 1);
    $('search-limit').textContent = state.searchLimit;
  });
  $('search-submit').addEventListener('click', submitSearch);
  $('search-query').addEventListener('keydown', (event) => {
    if (event.key === 'Enter') submitSearch();
  });
}

/* ============ Arranque ============ */

function init() {
  $('host-chip').textContent = location.host;
  window.addEventListener('hashchange', navigate);
  navigate();
  setupIngest();
  setupSearch();
  renderActivity();
  loadHealth();
  loadOverview();
  loadAgents();
  loadSources();
  setInterval(loadHealth, 30000);
}

init();
