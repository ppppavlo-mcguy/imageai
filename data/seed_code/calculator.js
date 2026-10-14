// instruction: write a javascript calculator
const display = document.querySelector('#display');
let current = '';

function press(value) {
  if (value === 'C') {
    current = '';
  } else if (value === '=') {
    current = String(eval(current));
  } else {
    current += value;
  }
  display.textContent = current || '0';
}

for (const button of document.querySelectorAll('button')) {
  button.addEventListener('click', () => press(button.textContent));
}
