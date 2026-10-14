# instruction: write javascript that validates an email form
const form = document.getElementById('form');
const email = document.getElementById('email');

form.addEventListener('submit', ev => {
  ev.preventDefault();
  const value = email.value.trim();
  if (!/^[^@\s]+@[^@\s]+\.[a-z]{2,}$/i.test(value)) {
    email.setCustomValidity('please enter a valid email address');
    email.reportValidity();
    return;
  }
  email.setCustomValidity('');
  form.classList.add('sent');
});
