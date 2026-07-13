const API_BASE = '';
let sessionId = generateId();
let isStreaming = false;
let currentFeedbackBar = null;

function generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
}

function $(sel) { return document.querySelector(sel); }
function $$(sel) { return document.querySelectorAll(sel); }

function showToast(msg) {
    const t = $('#toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 3000);
}

async function api(path, opts = {}) {
    const url = `${API_BASE}${path}`;
    const res = await fetch(url, {
        headers: { 'Accept': 'application/json', ...(opts.headers || {}) },
        ...opts
    });
    if (!res.ok) {
        let err = `HTTP ${res.status}`;
        try { const j = await res.json(); err = j.detail || JSON.stringify(j); } catch {}
        throw new Error(err);
    }
    if (opts.raw) return res;
    const ct = res.headers.get('content-type') || '';
    return ct.includes('application/json') ? res.json() : res.text();
}

function escapeHtml(s) {
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
}

function renderMath(element) {
    if (typeof renderMathInElement === 'undefined') return;
    try {
        renderMathInElement(element, {
            delimiters: [
                {left: '$$', right: '$$', display: true},
                {left: '$', right: '$', display: false},
                {left: '\\(', right: '\\)', display: false},
                {left: '\\[', right: '\\]', display: true}
            ],
            throwOnError: false
        });
    } catch (e) {
        console.error('KaTeX render failed:', e);
    }
}

function markdownToHtml(md) {
    let html = escapeHtml(md);
    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (_, lang, code) => {
        return `<pre><code class="language-${lang || ''}">${escapeHtml(code.trim())}</code></pre>`;
    });
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    html = html.replace(/^#{1,6}\s+(.+)$/gm, (_, t) => `<strong>${t}</strong>`);
    html = html.replace(/\n/g, '<br>');
    return html;
}

function addMessage(role, content, meta) {
    const container = $('#chat-container');
    const welcome = $('#welcome');
    if (welcome) welcome.remove();

    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    const avatarText = role === 'user' ? '我' : 'AI';
    const metaHtml = meta ? `<div class="message-meta">${meta}</div>` : '';
    msg.innerHTML = `
        <div class="avatar ${role}">${avatarText}</div>
        <div class="message-content">
            <div class="msg-body">${role === 'assistant' ? '' : markdownToHtml(content)}</div>
            ${metaHtml}
        </div>
    `;
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
    renderMath(msg.querySelector('.message-content'));
    return msg;
}

function addTyping() {
    const container = $('#chat-container');
    const el = document.createElement('div');
    el.className = 'message assistant typing-msg';
    el.innerHTML = `
        <div class="avatar assistant">AI</div>
        <div class="message-content">
            <div class="typing-indicator"><span></span><span></span><span></span></div>
        </div>
    `;
    container.appendChild(el);
    container.scrollTop = container.scrollHeight;
    return el;
}

function removeTyping() {
    const t = $('.typing-msg');
    if (t) t.remove();
}

function createFeedbackBar(question, answer) {
    const bar = document.createElement('div');
    bar.className = 'feedback-bar';
    bar.innerHTML = `
        <span style="font-size:12px;color:var(--text-secondary);">这个回答对您有帮助吗？</span>
        <button class="feedback-btn correct" data-value="correct">✓ 正确</button>
        <button class="feedback-btn wrong" data-value="wrong">✗ 错误</button>
    `;
    bar.querySelectorAll('.feedback-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const value = btn.dataset.value;
            try {
                await api('/api/rag/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, answer, feedback: value })
                });
                bar.innerHTML = '<span style="font-size:12px;color:var(--accent);">感谢您的反馈！</span>';
            } catch (e) {
                showToast('反馈提交失败: ' + e.message);
            }
        });
    });
    return bar;
}

function hideCurrentFeedback() {
    if (currentFeedbackBar) {
        currentFeedbackBar.classList.add('hidden');
        currentFeedbackBar = null;
    }
}

async function sendQuestion() {
    const input = $('#question-input');
    const btn = $('#send-btn');
    const question = input.value.trim();
    if (!question || isStreaming) return;

    hideCurrentFeedback();
    input.value = '';
    input.style.height = 'auto';
    btn.disabled = true;

    const mode = $('#mode-select').value || null;
    const stream = $('#stream-toggle').checked;
    const multimodal = $('#multimodal-toggle').checked;

    addMessage('user', question);
    const typing = addTyping();
    isStreaming = true;

    try {
        if (stream) {
            const res = await api('/api/rag/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, session_id: sessionId, mode, stream: true, multimodal }),
                raw: true
            });
            removeTyping();

            const msgEl = addMessage('assistant', '');
            const bodyEl = msgEl.querySelector('.msg-body');
            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let answerText = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                answerText += chunk;
                bodyEl.innerHTML = markdownToHtml(answerText);
                $('#chat-container').scrollTop = $('#chat-container').scrollHeight;
            }
            renderMath(bodyEl);

            const metaParts = [];
            if (mode) metaParts.push(`模式: ${mode}`);
            if (multimodal) metaParts.push('多模态');
            const metaEl = document.createElement('div');
            metaEl.className = 'message-meta';
            metaEl.textContent = metaParts.join(' | ') || '智能路由';
            msgEl.querySelector('.message-content').appendChild(metaEl);

            const fb = createFeedbackBar(question, answerText);
            msgEl.querySelector('.message-content').appendChild(fb);
            currentFeedbackBar = fb;
        } else {
            const data = await api('/api/rag/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question, session_id: sessionId, mode, stream: false, multimodal })
            });
            removeTyping();
            const answer = data.answer || '';
            const metaParts = [];
            if (data.mode_selection) {
                const s = data.mode_selection;
                metaParts.push(`难度: ${s.difficulty || '-'} | 复杂度: ${s.complexity || '-'} | 选中: ${s.mode || '-'}`);
            } else if (mode) {
                metaParts.push(`模式: ${mode}`);
            }
            if (multimodal) metaParts.push('多模态');
            const msgEl = addMessage('assistant', answer, metaParts.join(' | ') || '智能路由');

            const fb = createFeedbackBar(question, answer);
            msgEl.querySelector('.message-content').appendChild(fb);
            currentFeedbackBar = fb;
        }
    } catch (e) {
        removeTyping();
        addMessage('assistant', `请求出错: ${escapeHtml(e.message)}`);
    } finally {
        isStreaming = false;
        btn.disabled = !$('#question-input').value.trim();
    }
}

async function loadDocuments() {
    try {
        const data = await api('/api/rag/documents');
        const list = $('#doc-list');
        list.innerHTML = '';
        (data.items || []).forEach(doc => {
            const el = document.createElement('div');
            el.className = 'doc-item';
            const name = doc.file_name || doc.doc_id || '未知';
            el.innerHTML = `<span title="${escapeHtml(name)}">${escapeHtml(name)}</span>`;
            list.appendChild(el);
        });
    } catch (e) {
        console.error('load docs failed', e);
    }
}

async function uploadFiles(files) {
    if (!files.length) return;
    const fd = new FormData();
    files.forEach(f => fd.append('files', f));
    showToast(`正在上传 ${files.length} 个文件...`);
    try {
        await api('/api/rag/documents/batch', { method: 'POST', body: fd });
        showToast('上传成功');
        loadDocuments();
    } catch (e) {
        showToast('上传失败: ' + e.message);
    }
}

async function doIncremental() {
    showToast('正在执行增量更新...');
    try {
        await api('/api/rag/incremental-update', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
        showToast('增量更新完成');
    } catch (e) {
        showToast('增量更新失败: ' + e.message);
    }
}

$('#send-btn').addEventListener('click', sendQuestion);
$('#question-input').addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendQuestion();
    }
});
$('#question-input').addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 200) + 'px';
    $('#send-btn').disabled = !this.value.trim() || isStreaming;
});

$('#new-chat-btn').addEventListener('click', () => {
    sessionId = generateId();
    $('#chat-container').innerHTML = `
        <div class="welcome" id="welcome">
            <h2>Graph RAG 智能问答系统</h2>
            <p>基于 LightRAG + Neo4j + Milvus 构建</p>
            <div class="welcome-tips">
                <div class="tip">📚 上传文档后自动索引</div>
                <div class="tip">🔍 支持多模式检索（naive/local/global/hybrid/mix）</div>
                <div class="tip">🧠 LLM 智能路由自动选择最佳检索策略</div>
                <div class="tip">🖼️ 支持多模态视觉问答</div>
            </div>
        </div>
    `;
    hideCurrentFeedback();
    showToast('已开启新对话');
});

const uploadArea = $('#upload-area');
const fileInput = $('#file-input');
uploadArea.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('dragover'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('dragover'));
uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    uploadFiles(Array.from(e.dataTransfer.files));
});
fileInput.addEventListener('change', () => uploadFiles(Array.from(fileInput.files)));

$('#incremental-btn').addEventListener('click', doIncremental);

loadDocuments();
