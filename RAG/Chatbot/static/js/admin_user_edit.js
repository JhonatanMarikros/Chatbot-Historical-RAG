// static/js/admin_user_edit.js
(function () {
  const btn = document.getElementById('btnDeleteUser');
  const formDelUser = document.getElementById('f_del_user');
  if (!btn || !formDelUser) return;

  const msg = btn.dataset.msgDelUser
    || 'Hapus user ini beserta SELURUH datanya (termasuk semua query, log, dan evaluasi)?';

  btn.addEventListener('click', () => {
    if (confirm(msg)) {
      formDelUser.submit();
    }
    // Cancel -> tidak melakukan apa pun.
  });
})();
