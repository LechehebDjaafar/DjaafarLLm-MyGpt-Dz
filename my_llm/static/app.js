// ════════════════════════════════
//  جعفر AI — app.js
// ════════════════════════════════

const chatBox    = document.getElementById('chat-box');
const welcome    = document.getElementById('welcome');
const historyEl  = document.getElementById('history-list');
const titleEl    = document.getElementById('chat-title');
const inputEl    = document.getElementById('user-input');
const sendBtn    = document.getElementById('send-btn');
const modelPick  = document.getElementById('model-picker');

let sessions  = JSON.parse(localStorage.getItem('jfr_sessions') || '{}');
let settings  = JSON.parse(localStorage.getItem('jfr_settings') || '{"model":"v2","temp":0.6,"tokens":60}');
let currentId = null;
let busy      = false;

// ── Init ──
applySettings();
renderHistory();

// ── Suggestions ──
document.querySelectorAll('.sug-btn').forEach(b => {
  b.addEventListener('click', () => { inputEl.value = b.textContent; sendMessage(); });
});

// ── New Chat ──
document.getElementById('new-chat-btn').addEventListener('click', newChat);

// ── Clear All ──
document.getElementById('clear-btn').addEventListener('click', () => {
  if (!confirm('حذف كل المحادثات؟')) return;
  sessions = {}; currentId = null;
  save(); renderHistory(); showWelcome();
});

// ── Settings ──
document.getElementById('settings-btn').addEventListener('click',  openSettings);
document.getElementById('close-modal').addEventListener('click',   closeSettings);
document.getElementById('save-settings-btn').addEventListener('click', saveSettings);
document.getElementById('modal-overlay').addEventListener('click', e => {
  if (e.target === document.getElementById('modal-overlay')) closeSettings();
});

document.getElementById('s-temp').addEventListener('input', function() {
  document.getElementById('s-temp-val').textContent = this.value;
});
document.getElementById('s-tokens').addEventListener('input', function() {
  document.getElementById('s-tokens-val').textContent = this.value;
});

// ── Send ──
sendBtn.addEventListener('click', sendMessage);
inputEl.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
inputEl.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.min(this.scrollHeight, 150) + 'px';
});

// model picker in topbar
modelPick.addEventListener('change', () => {
  settings.model = modelPick.value;
  document.getElementById('s-model').value = settings.model;
  localStorage.setItem('jfr_settings', JSON.stringify(settings));
});

// ════════════════ Functions ════════════════

function newChat() {
  currentId = Date.now().toString();
  sessions[currentId] = { title: 'دردشة جديدة', messages: [] };
  save(); renderHistory(); showWelcome();
  titleEl.textContent = 'دردشة جديدة';
}

function showWelcome() {
  chatBox.innerHTML = '';
  chatBox.appendChild(welcome);
  welcome.style.display = 'flex';
}

function renderHistory() {
  historyEl.innerHTML = '';
  const ids = Object.keys(sessions).reverse();
  if (!ids.length) {
    historyEl.innerHTML = '<div class="h-empty">لا توجد محادثات بعد</div>';
    return;
  }
  ids.forEach(id => {
    const s   = sessions[id];
    const div = document.createElement('div');
    div.className = 'h-item' + (id === currentId ? ' active' : '');
    div.innerHTML = `
      <span>💬</span>
      <span class="h-title">${s.title}</span>
      <button class="h-del" data-id="${id}" title="حذف">✕</button>`;
    div.addEventListener('click', e => {
      if (!e.target.classList.contains('h-del')) loadSession(id);
    });
    div.querySelector('.h-del').addEventListener('click', () => deleteSession(id));
    historyEl.appendChild(div);
  });
}

function loadSession(id) {
  currentId = id;
  renderHistory();
  renderMessages();
}

function deleteSession(id) {
  delete sessions[id];
  if (currentId === id) { currentId = null; }
  save(); renderHistory();
  if (!currentId) showWelcome();
  else renderMessages();
}

function renderMessages() {
  chatBox.innerHTML = '';
  const msgs = sessions[currentId]?.messages || [];
  titleEl.textContent = sessions[currentId]?.title || 'دردشة';
  msgs.forEach(m => appendBubble(m.role, m.text, m.source, false));
  chatBox.scrollTop = chatBox.scrollHeight;
}

function appendBubble(role, text, source = '', animate = true) {
  welcome.style.display = 'none';
  const wrap = document.createElement('div');
  wrap.className = 'msg-wrap ' + role;

  const label  = role === 'user' ? 'أنت' : '🤖 جعفر AI';
  const bubble = document.createElement('div');
  bubble.className = 'bubble ' + role;
  bubble.textContent = text;

  if (source) {
    const src = document.createElement('div');
    src.className = 'msg-source';
    src.textContent = '⚡ ' + source;
    bubble.appendChild(src);
  }

  wrap.innerHTML = `<div class="msg-label">${label}</div>`;
  wrap.appendChild(bubble);
  chatBox.appendChild(wrap);
  if (animate) chatBox.scrollTop = chatBox.scrollHeight;
  return bubble;
}

// Typewriter effect
function typewrite(element, text, speed = 18) {
  return new Promise(resolve => {
    let i = 0;
    element.textContent = '';
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    element.appendChild(cursor);

    const tick = () => {
      if (i < text.length) {
        element.insertBefore(document.createTextNode(text[i]), cursor);
        i++;
        chatBox.scrollTop = chatBox.scrollHeight;
        setTimeout(tick, speed);
      } else {
        cursor.remove();
        resolve();
      }
    };
    tick();
  });
}

async function sendMessage() {
  const msg = inputEl.value.trim();
  if (!msg || busy) return;

  if (!currentId) newChat();

  // Set title
  if (sessions[currentId].messages.length === 0) {
    sessions[currentId].title = msg.slice(0, 30) + (msg.length > 30 ? '…' : '');
  }

  sessions[currentId].messages.push({ role: 'user', text: msg, source: '' });
  save(); renderHistory();
  appendBubble('user', msg);
  inputEl.value = ''; inputEl.style.height = 'auto';

  busy = true; sendBtn.disabled = true;

  // Typing indicator
  const typingWrap = document.createElement('div');
  typingWrap.className = 'msg-wrap bot';
  typingWrap.innerHTML = `
    <div class="msg-label">🤖 جعفر AI</div>
    <div class="typing-bubble">
      <span></span><span></span><span></span>
    </div>`;
  chatBox.appendChild(typingWrap);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const res  = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: msg,
        model:   settings.model,
        temp:    settings.temp,
        tokens:  settings.tokens
      })
    });

    const data = await res.json();
    typingWrap.remove();

    // Append bubble then typewrite
    const bubble = appendBubble('bot', '', data.source || '');
    await typewrite(bubble, data.response, 22);

    // إضافة source بعد الكتابة
    if (data.source) {
      const src = document.createElement('div');
      src.className = 'msg-source';
      src.textContent = '⚡ ' + data.source;
      bubble.appendChild(src);
    }

    sessions[currentId].messages.push({
      role: 'bot', text: data.response, source: data.source || ''
    });
    save();

  } catch (err) {
    typingWrap.remove();
    appendBubble('bot', '❌ خطأ في الاتصال بالخادم. تأكد أن Flask يعمل.');
  }

  busy = false; sendBtn.disabled = false;
  inputEl.focus();
}

// ════ Settings ════
function openSettings() {
  document.getElementById('s-model').value      = settings.model;
  document.getElementById('s-temp').value        = settings.temp;
  document.getElementById('s-tokens').value      = settings.tokens;
  document.getElementById('s-temp-val').textContent   = settings.temp;
  document.getElementById('s-tokens-val').textContent = settings.tokens;
  document.getElementById('modal-overlay').classList.add('show');
}
function closeSettings() {
  document.getElementById('modal-overlay').classList.remove('show');
}
function saveSettings() {
  settings.model  = document.getElementById('s-model').value;
  settings.temp   = parseFloat(document.getElementById('s-temp').value);
  settings.tokens = parseInt(document.getElementById('s-tokens').value);
  modelPick.value = settings.model;
  localStorage.setItem('jfr_settings', JSON.stringify(settings));
  closeSettings();
}
function applySettings() {
  modelPick.value = settings.model;
}

function save() { localStorage.setItem('jfr_sessions', JSON.stringify(sessions)); }