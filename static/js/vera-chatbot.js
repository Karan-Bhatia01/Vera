/**
 * Vera RAG Chatbot Widget
 * Injects a floating chatbot powered by DeepSeek R1 8B + BGE-Large RAG.
 * Auto-detects the current dataset from the page URL.
 */
(function () {
    'use strict';

    // ── State ──────────────────────────────────────────────────────────────────
    let history  = [];
    let isOpen   = false;
    let loading  = false;

    // Try to detect current filename from URL params
    function getFilename() {
        const p = new URLSearchParams(window.location.search);
        return p.get('filename') || null;
    }

    // ── Build DOM ──────────────────────────────────────────────────────────────
    function buildWidget() {
        const widget = document.createElement('div');
        widget.id    = 'vera-chat-widget';
        widget.innerHTML = `
            <div id="vera-chat-panel">
                <div class="vera-panel-header">
                    <div class="vera-avatar">✨</div>
                    <div>
                        <h4>Vera Assistant</h4>
                        <p>Powered by DeepSeek R1 · BGE-Large RAG</p>
                    </div>
                    <button class="vera-panel-close" onclick="veraChatClose()" title="Close">×</button>
                </div>

                <div class="vera-messages" id="vera-msgs">
                    <div class="vera-msg bot">
                        👋 Hi! I'm Vera. Ask me anything about your datasets, insights, or ML results.
                    </div>
                </div>

                <div class="vera-suggestions" id="vera-suggestions">
                    <button class="vera-suggestion-btn" onclick="veraSuggest(this)">What insights were found in this dataset?</button>
                    <button class="vera-suggestion-btn" onclick="veraSuggest(this)">Which feature impacts the output most?</button>
                    <button class="vera-suggestion-btn" onclick="veraSuggest(this)">What are the data quality issues?</button>
                </div>

                <div class="vera-input-area">
                    <textarea id="vera-input" rows="1" placeholder="Ask about your data…"
                        onkeydown="veraKeydown(event)" oninput="veraResize(this)"></textarea>
                    <button id="vera-send-btn" onclick="veraSend()" title="Send">➤</button>
                </div>
            </div>

            <button id="vera-chat-toggle" onclick="veraToggle()" title="Vera AI Assistant">✨</button>
        `;
        document.body.appendChild(widget);
    }

    // ── Toggle ─────────────────────────────────────────────────────────────────
    window.veraToggle = function () {
        const panel = document.getElementById('vera-chat-panel');
        isOpen = !isOpen;
        panel.classList.toggle('open', isOpen);
        if (isOpen) {
            setTimeout(() => document.getElementById('vera-input')?.focus(), 120);
        }
    };

    window.veraChatClose = function () {
        isOpen = false;
        document.getElementById('vera-chat-panel').classList.remove('open');
    };

    // ── Suggest ────────────────────────────────────────────────────────────────
    window.veraSuggest = function (btn) {
        const q = btn.textContent.trim();
        document.getElementById('vera-input').value = q;
        // Hide suggestions after first use
        document.getElementById('vera-suggestions').style.display = 'none';
        veraSend();
    };

    // ── Resize textarea ────────────────────────────────────────────────────────
    window.veraResize = function (el) {
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 100) + 'px';
    };

    // ── Keyboard: Enter to send, Shift+Enter for newline ──────────────────────
    window.veraKeydown = function (e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            veraSend();
        }
    };

    // ── Send ───────────────────────────────────────────────────────────────────
    window.veraSend = async function () {
        if (loading) return;
        const input = document.getElementById('vera-input');
        const query = input.value.trim();
        if (!query) return;

        // Clear input
        input.value   = '';
        input.style.height = 'auto';

        // Append user message
        appendMsg(query, 'user');

        // Hide suggestion buttons
        document.getElementById('vera-suggestions').style.display = 'none';

        // Show typing indicator
        const typingEl = showTyping();
        loading = true;
        document.getElementById('vera-send-btn').disabled = true;

        try {
            const body = { query, history, filename: getFilename() };
            const res  = await fetch('/api/chat', {
                method:  'POST',
                headers: { 'Content-Type': 'application/json' },
                body:    JSON.stringify(body),
            });
            const data = await res.json();

            typingEl.remove();

            const answer  = data.answer  || 'No response.';
            const sources = data.sources || [];

            appendMsg(answer, 'bot', sources);

            // Update history (keep last 6 turns)
            history.push({ role: 'user',      content: query  });
            history.push({ role: 'assistant', content: answer });
            if (history.length > 12) history = history.slice(-12);

        } catch (err) {
            typingEl.remove();
            appendMsg('⚠️ Network error. Please try again.', 'bot');
        }

        loading = false;
        document.getElementById('vera-send-btn').disabled = false;
        input.focus();
    };

    // ── Helpers ────────────────────────────────────────────────────────────────
    function appendMsg(text, role, sources) {
        const msgs = document.getElementById('vera-msgs');
        const div  = document.createElement('div');
        div.className = `vera-msg ${role}`;

        // Convert markdown-ish line breaks and bold
        const formatted = text
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\n/g, '<br>');
        div.innerHTML = formatted;

        if (sources && sources.length) {
            const src = document.createElement('div');
            src.className = 'vera-sources';
            src.textContent = '📂 Sources: ' + sources.join(', ');
            div.appendChild(src);
        }

        msgs.appendChild(div);
        msgs.scrollTop = msgs.scrollHeight;
    }

    function showTyping() {
        const msgs = document.getElementById('vera-msgs');
        const el   = document.createElement('div');
        el.className = 'vera-typing';
        el.innerHTML = '<span></span><span></span><span></span>';
        msgs.appendChild(el);
        msgs.scrollTop = msgs.scrollHeight;
        return el;
    }

    // ── Init ───────────────────────────────────────────────────────────────────
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', buildWidget);
    } else {
        buildWidget();
    }
})();
