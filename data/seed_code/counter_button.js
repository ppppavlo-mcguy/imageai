# instruction: write javascript for a counter button
const button = document.getElementById('counter');
let count = 0;

button.textContent = `Clicked ${count} times`;
button.addEventListener('click', () => {
  count += 1;
  button.textContent = `Clicked ${count} times`;
});
