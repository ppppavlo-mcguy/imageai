// instruction: build a javascript quiz app
const questions = [
  { q: 'What is 2 + 2?', a: ['3', '4', '5'], correct: 1 },
  { q: 'Which colour is the sky?', a: ['blue', 'red', 'green'], correct: 0 },
];
let index = 0;
let score = 0;

function show() {
  const item = questions[index];
  document.querySelector('#question').textContent = item.q;
  document.querySelector('#answers').innerHTML = item.a
    .map((text, i) => `<button data-i="${i}">${text}</button>`).join('');
}

document.querySelector('#answers').addEventListener('click', event => {
  if (Number(event.target.dataset.i) === questions[index].correct) score += 1;
  index += 1;
  if (index < questions.length) show();
  else document.querySelector('#question').textContent = `Score ${score}`;
});

show();
