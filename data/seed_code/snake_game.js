// instruction: make a snake game in javascript
const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
const size = 16;
let snake = [{ x: 8, y: 8 }];
let dir = { x: 1, y: 0 };
let food = { x: 4, y: 4 };
let score = 0;

function reset() {
  snake = [{ x: 8, y: 8 }];
  dir = { x: 1, y: 0 };
  score = 0;
}

function step() {
  const head = { x: snake[0].x + dir.x, y: snake[0].y + dir.y };
  if (head.x < 0 || head.y < 0 || head.x >= size || head.y >= size) return reset();
  snake.unshift(head);
  if (head.x === food.x && head.y === food.y) {
    score += 1;
    food = { x: Math.floor(Math.random() * size), y: Math.floor(Math.random() * size) };
  } else {
    snake.pop();
  }
}

function draw() {
  ctx.fillStyle = '#101014';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#6ee7a8';
  for (const part of snake) ctx.fillRect(part.x * 16, part.y * 16, 15, 15);
  ctx.fillStyle = '#e0812a';
  ctx.fillRect(food.x * 16, food.y * 16, 15, 15);
}

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowUp') dir = { x: 0, y: -1 };
  if (e.key === 'ArrowDown') dir = { x: 0, y: 1 };
  if (e.key === 'ArrowLeft') dir = { x: -1, y: 0 };
  if (e.key === 'ArrowRight') dir = { x: 1, y: 0 };
});

setInterval(() => { step(); draw(); }, 120);
