(function () {
  const chatForm  = document.getElementById('chatForm');
  const userInput = document.getElementById('userInput');
  const chatArea  = document.getElementById('chatArea');
  const submitBtn = document.getElementById('submitBtn');
  const root      = document.getElementById('chatRoot');

  // Ambil konfigurasi dari data-attributes
  const GET_RESPONSE_URL = chatForm?.dataset?.endpoint || '/get_response';
  const LOGIN_URL        = chatForm?.dataset?.login || '/login';
  const USERNAME         = (root?.dataset?.username || 'User').trim();

  function appendUser(msg){
    const row = document.createElement('div'); 
    row.className = 'chat-row'; 
    row.style.justifyContent = 'flex-end';
    row.innerHTML = `
      <div class="bubble"></div>
      <div style="text-align:center">
        <div class="avatar">👤</div>
        <span class="muted"></span>
      </div>`;
    row.querySelector('.bubble').textContent = msg;
    row.querySelector('.muted').textContent = USERNAME;
    chatArea.appendChild(row);
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  function appendBot(msg){
    const row = document.createElement('div'); 
    row.className = 'chat-row';
    row.innerHTML = `
      <div style="text-align:center">
        <div class="avatar">🤖</div>
        <div class="chat-label">Chatbot</div>
      </div>
      <div class="bubble"></div>`;
    row.querySelector('.bubble').textContent = msg;
    chatArea.appendChild(row);
    chatArea.scrollTop = chatArea.scrollHeight;
  }

  // === Loading/Typing bubble ===
  function appendTyping(){
    const row = document.createElement('div'); 
    row.className = 'chat-row typing-row';
    row.innerHTML = `
      <div style="text-align:center">
        <div class="avatar">🤖</div>
        <div class="chat-label">Chatbot</div>
      </div>
      <div class="bubble typing">
        <span class="dot"></span><span class="dot"></span><span class="dot"></span>
      </div>`;
    chatArea.appendChild(row);
    chatArea.scrollTop = chatArea.scrollHeight;
    return row; // untuk dihapus/diganti nanti
  }

  function lockForm(){
    userInput.disabled = true;
    submitBtn.disabled = true;
    submitBtn.dataset.oldText = submitBtn.textContent;
    submitBtn.textContent = 'Loading…';
  }
  function unlockForm(){
    userInput.disabled = false;
    submitBtn.disabled = false;
    submitBtn.textContent = submitBtn.dataset.oldText || 'Submit';
    userInput.focus();
  }

  if (chatForm) {
    chatForm.addEventListener('submit', async (e)=>{
      e.preventDefault();
      const text = (userInput.value || '').trim();
      if(!text) return;

      userInput.value = '';
      appendUser(text);

      // tampilkan gelembung typing & kunci form
      const typingRow = appendTyping();
      lockForm();

      try{
        const resp = await fetch(GET_RESPONSE_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
          body: `user_message=${encodeURIComponent(text)}`
        });

        // jika server redirect (mis. belum login), ikuti
        const ct = resp.headers.get('content-type') || '';
        if (resp.redirected || (!ct.includes('application/json') && resp.ok)) {
          window.location.href = resp.url || LOGIN_URL;
          return;
        }

        if (!resp.ok) {
          const b = typingRow.querySelector('.bubble');
          b.classList.remove('typing');
          b.textContent = 'Terjadi kesalahan server.';
          return;
        }

        const data = await resp.json();

        // ganti typing dengan jawaban bot
        typingRow.remove();
        appendBot(data.response || '(kosong)');
      } catch (err) {
        const b = typingRow.querySelector('.bubble');
        b.classList.remove('typing');
        b.textContent = 'Gagal terhubung ke server.';
      } finally {
        unlockForm();
      }
    });
  }
})();
