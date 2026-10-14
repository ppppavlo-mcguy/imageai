# instruction: write javascript that toggles between light and dark mode
const root = document.documentElement;
let dark = localStorage.getItem('theme') === 'dark';

function apply() {
  root.style.background = dark ? '#14161a' : '#ffffff';
  root.style.color = dark ? '#e8ecf1' : '#111111';
  localStorage.setItem('theme', dark ? 'dark' : 'light');
}

document.getElementById('toggle').addEventListener('click', () => {
  dark = !dark;
  apply();
});

apply();
