document.addEventListener('DOMContentLoaded', () => {
  const toggleBtn = document.getElementById('chatbot-toggle-btn');
  const header = document.getElementById('chatbot-header');
  const body = document.getElementById('chatbot-body');
  const sendBtn = document.getElementById('chat-send-btn');
  const input = document.getElementById('chat-input');
  const messages = document.getElementById('chat-messages');

  if (!toggleBtn || !header || !body || !sendBtn || !input || !messages) return;

  const setOpen = (open) => {
    body.style.display = open ? 'flex' : 'none';
    toggleBtn.innerHTML = open ? '&#9660;' : '&#9650;';
    if (open) messages.scrollTop = messages.scrollHeight;
  };

  header.addEventListener('click', (event) => {
    if (event.target === input || event.target === sendBtn) return;
    setOpen(body.style.display === 'none');
  });

  const appendMessage = (role, text) => {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;
    msg.textContent = text;
    messages.appendChild(msg);
    messages.scrollTop = messages.scrollHeight;
  };

  const appendThinking = () => {
    const msg = document.createElement('div');
    msg.className = 'message assistant';
    msg.textContent = 'Thinking...';
    messages.appendChild(msg);
    messages.scrollTop = messages.scrollHeight;
    return msg;
  };

  const sendMessage = async (overrideText = null) => {
    const text = (overrideText || input.value).trim();
    if (!text) return;

    appendMessage('user', text);
    if (!overrideText) input.value = '';
    input.disabled = true;
    sendBtn.disabled = true;

    const thinking = appendThinking();

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, message: text })
      });
      thinking.remove();

      if (!res.ok) {
        appendMessage('assistant', 'Error communicating with Copilot. Please try again.');
        return;
      }

      const data = await res.json();
      appendMessage('assistant', data.response || 'No response.');
    } catch (err) {
      thinking.remove();
      appendMessage('assistant', `Network error: ${err.message}`);
    } finally {
      input.disabled = false;
      sendBtn.disabled = false;
      input.focus();
    }
  };

  sendBtn.addEventListener('click', () => sendMessage());
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') sendMessage();
  });

  window.sendQuick = (question) => {
    setOpen(true);
    sendMessage(question);
  };

  setOpen(false);
});
