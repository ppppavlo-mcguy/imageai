// instruction: make a tic tac toe game
const board = Array(9).fill('');
let turn = 'X';

function winner() {
  const lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]];
  for (const [a, b, c] of lines) {
    if (board[a] && board[a] === board[b] && board[a] === board[c]) return board[a];
  }
  return board.includes('') ? null : 'draw';
}

function play(index) {
  if (board[index] || winner()) return;
  board[index] = turn;
  turn = turn === 'X' ? 'O' : 'X';
  render();
}
