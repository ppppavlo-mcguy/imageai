# instruction: write javascript for a todo list with add and remove
const list = document.getElementById('list');
const input = document.getElementById('input');

function add(text) {
  if (!text) return;
  const li = document.createElement('li');
  li.textContent = text;
  const remove = document.createElement('button');
  remove.textContent = 'x';
  remove.addEventListener('click', () => li.remove());
  li.appendChild(remove);
  list.appendChild(li);
  input.value = '';
}

document.getElementById('add').addEventListener('click', () => add(input.value.trim()));
