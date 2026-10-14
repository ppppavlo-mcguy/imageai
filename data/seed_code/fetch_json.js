# instruction: write javascript that fetches json and renders it as a list
async function render(url, target) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const items = await res.json();
  const ul = document.createElement('ul');
  for (const item of items) {
    const li = document.createElement('li');
    li.textContent = item.name ?? String(item);
    ul.appendChild(li);
  }
  target.replaceChildren(ul);
}

render('/api/items', document.getElementById('app')).catch(err => console.error(err));
