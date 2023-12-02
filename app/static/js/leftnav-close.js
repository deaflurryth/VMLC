const menuToggle = document.getElementById('menu-toggle');
const leftNav = document.getElementById('left-nav');

menuToggle.addEventListener('click', function () {
    menuToggle.classList.toggle('hidden');
    leftNav.classList.toggle('hidden');
});