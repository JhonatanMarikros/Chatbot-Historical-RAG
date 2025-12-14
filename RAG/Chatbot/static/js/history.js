// toggle evaluasi (tanpa ubah style)
document.addEventListener('click', function (e) {
  const btn = e.target.closest('.btn-mini');
  if (!btn) return;
  const id = btn.getAttribute('data-target');
  const row = document.getElementById(id);
  if (!row) return;

  const hidden = (row.style.display === '' || row.style.display === 'none');
  row.style.display = hidden ? 'table-row' : 'none';
  btn.setAttribute('aria-expanded', String(hidden));

  // ubah ikon
  const chev = btn.querySelector('.chev');
  if (chev) chev.textContent = hidden ? '∧' : '∨';
});
