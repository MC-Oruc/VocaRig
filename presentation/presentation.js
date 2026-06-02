document.addEventListener('DOMContentLoaded', () => {
  const slides = document.querySelectorAll('.slide');
  const totalSlides = slides.length;
  let currentSlideIndex = 0;

  // Controls
  const btnPrev = document.getElementById('btn-prev');
  const btnNext = document.getElementById('btn-next');
  const progressBar = document.querySelector('.progress-bar');
  const slideCounterVal = document.getElementById('counter-val');
  const themeToggle = document.getElementById('theme-toggle');
  const fullscreenToggle = document.getElementById('fullscreen-toggle');
  const dotNav = document.createElement('div');
  dotNav.className = 'slide-dots';
  dotNav.setAttribute('aria-label', 'Slayt gezgini');
  document.body.appendChild(dotNav);

  slides.forEach((slide, index) => {
    const dot = document.createElement('button');
    dot.type = 'button';
    dot.className = 'slide-dot';
    dot.setAttribute('aria-label', `${index + 1}. slayta git`);
    dot.addEventListener('click', () => goToSlide(index));
    dotNav.appendChild(dot);
  });

  // Load Saved Theme (allow URL override for PDF generation without saving to localStorage)
  const urlParams = new URLSearchParams(window.location.search);
  const themeParam = urlParams.get('theme');
  const savedTheme = themeParam || localStorage.getItem('vocarig-theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);

  // Initialize
  const requestedSlide = Number(new URLSearchParams(window.location.search).get('slide'));
  const hashSlide = Number((window.location.hash.match(/slide-(\d+)/) || [])[1]);
  const initialSlide = Number.isInteger(requestedSlide) && requestedSlide > 0
    ? requestedSlide - 1
    : Number.isInteger(hashSlide) && hashSlide > 0
      ? hashSlide - 1
      : 0;
  goToSlide(Math.min(Math.max(initialSlide, 0), totalSlides - 1));

  function goToSlide(index) {
    if (index < 0 || index >= totalSlides) return;

    slides.forEach((slide) => {
      slide.classList.remove('active');
    });

    slides[index].classList.add('active');
    currentSlideIndex = index;
    if (window.location.hash !== `#slide-${index + 1}`) {
      history.replaceState(null, '', `#slide-${index + 1}`);
    }

    // Update progress
    const progressPercent = ((index + 1) / totalSlides) * 100;
    progressBar.style.width = `${progressPercent}%`;
    slideCounterVal.textContent = `${index + 1} / ${totalSlides}`;
    dotNav.querySelectorAll('.slide-dot').forEach((dot, dotIndex) => {
      dot.classList.toggle('active', dotIndex === index);
    });

    // Button states
    btnPrev.disabled = index === 0;

    if (index === totalSlides - 1) {
      btnNext.querySelector('span').textContent = 'Başa Dön';
    } else {
      btnNext.querySelector('span').textContent = 'Sonraki';
    }

    // Trigger SVG animations on active slide
    triggerSvgAnimations(slides[index]);
  }

  function nextSlide() {
    if (currentSlideIndex === totalSlides - 1) {
      goToSlide(0);
    } else {
      goToSlide(currentSlideIndex + 1);
    }
  }

  function prevSlide() {
    if (currentSlideIndex > 0) {
      goToSlide(currentSlideIndex - 1);
    }
  }

  function triggerSvgAnimations(activeSlide) {
    const flowPaths = activeSlide.querySelectorAll('.svg-flow-path');
    flowPaths.forEach(path => {
      path.style.animation = 'none';
      path.offsetHeight;
      path.style.animation = '';
    });

    const pulseNodes = activeSlide.querySelectorAll('.svg-node');
    pulseNodes.forEach((node, i) => {
      node.classList.remove('pulse-node');
      setTimeout(() => {
        node.classList.add('pulse-node');
      }, i * 120);
    });
  }

  // Event Listeners
  btnNext.addEventListener('click', nextSlide);
  btnPrev.addEventListener('click', prevSlide);

  // Keyboard
  document.addEventListener('keydown', (e) => {
    switch (e.key) {
      case 'ArrowRight':
      case 'ArrowUp':
      case ' ':
      case 'Enter':
      case 'PageDown':
        e.preventDefault();
        nextSlide();
        break;
      case 'ArrowLeft':
      case 'ArrowDown':
      case 'Backspace':
      case 'PageUp':
        e.preventDefault();
        prevSlide();
        break;
      case 'Home':
        e.preventDefault();
        goToSlide(0);
        break;
      case 'End':
        e.preventDefault();
        goToSlide(totalSlides - 1);
        break;
    }
  });

  // Fare Tekerleği ile Slayt Geçişi (Wheel Navigation)
  let lastWheelTime = 0;
  const wheelCooldown = 600; // ms (hızlı tekerlek hareketlerini engellemek için bekleme süresi)

  document.addEventListener('wheel', (e) => {
    const now = Date.now();
    if (now - lastWheelTime < wheelCooldown) return;

    if (e.deltaY > 0) {
      // Aşağı kaydırma -> Sonraki slayt
      nextSlide();
      lastWheelTime = now;
    } else if (e.deltaY < 0) {
      // Yukarı kaydırma -> Önceki slayt
      prevSlide();
      lastWheelTime = now;
    }
  }, { passive: true });

  // Touch support for mobile
  let touchStartX = 0;
  let touchEndX = 0;
  document.addEventListener('touchstart', (e) => { touchStartX = e.changedTouches[0].screenX; }, false);
  document.addEventListener('touchend', (e) => {
    touchEndX = e.changedTouches[0].screenX;
    const diff = touchStartX - touchEndX;
    if (Math.abs(diff) > 60) {
      if (diff > 0) nextSlide();
      else prevSlide();
    }
  }, false);

  // Theme Toggle
  themeToggle.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('vocarig-theme', newTheme);
    updateThemeIcon(newTheme);
  });

  function updateThemeIcon(theme) {
    if (theme === 'dark') {
      themeToggle.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <circle cx="12" cy="12" r="5"></circle>
          <line x1="12" y1="1" x2="12" y2="3"></line>
          <line x1="12" y1="21" x2="12" y2="23"></line>
          <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
          <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
          <line x1="1" y1="12" x2="3" y2="12"></line>
          <line x1="21" y1="12" x2="23" y2="12"></line>
          <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
          <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
        </svg>`;
    } else {
      themeToggle.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path>
        </svg>`;
    }
  }

  // Fullscreen Toggle
  fullscreenToggle.addEventListener('click', () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen().catch(err => {
        console.error(`Tam ekran moduna geçilemedi: ${err.message}`);
      });
    } else {
      document.exitFullscreen();
    }
  });

  document.addEventListener('fullscreenchange', () => {
    const icon = document.fullscreenElement
      ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14h6v6m10-6h-6v6M4 10h6V4m10 6h-6V4"></path></svg>`
      : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M8 3H5a2 2 0 0 0-2 2v3m18 0V5a2 2 0 0 0-2-2h-3m0 18h3a2 2 0 0 0 2-2v-3M3 16v3a2 2 0 0 0 2 2h3"></path></svg>`;
    fullscreenToggle.innerHTML = icon;
  });

  // Kopyalama ve Kesme İşlemlerini Engelleme
  document.addEventListener('copy', (e) => e.preventDefault());
  document.addEventListener('cut', (e) => e.preventDefault());

  // ─── Laser Pointer Logic ───
  const canvas = document.getElementById('laser-pointer-canvas');
  const ctx = canvas.getContext('2d');
  let points = [];
  let isDrawing = false;

  function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);

  // Mouse event handlers
  document.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return; // Only trigger for left-click
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    points = [{ x: e.clientX - rect.left, y: e.clientY - rect.top, opacity: 1 }];
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    points.push({ x: e.clientX - rect.left, y: e.clientY - rect.top, opacity: 1 });
  });

  document.addEventListener('mouseup', () => {
    isDrawing = false;
  });

  document.addEventListener('mouseleave', () => {
    isDrawing = false;
  });

  // Touch event handlers
  document.addEventListener('touchstart', (e) => {
    if (e.touches.length === 1) {
      isDrawing = true;
      const touch = e.touches[0];
      const rect = canvas.getBoundingClientRect();
      points = [{ x: touch.clientX - rect.left, y: touch.clientY - rect.top, opacity: 1 }];
    }
  });

  document.addEventListener('touchmove', (e) => {
    if (!isDrawing) return;
    const touch = e.touches[0];
    const rect = canvas.getBoundingClientRect();
    points.push({ x: touch.clientX - rect.left, y: touch.clientY - rect.top, opacity: 1 });
  });

  document.addEventListener('touchend', () => {
    isDrawing = false;
  });

  // Helper to draw a smooth quadratic curve through points
  function drawCurve(strokeColorFn, lineWidthMultiplier) {
    if (points.length < 2) return;

    // Draw first straight segment from points[0] to the midpoint of points[0] and points[1]
    const p0 = points[0];
    const p1 = points[1];
    const mxInitial = (p0.x + p1.x) / 2;
    const myInitial = (p0.y + p1.y) / 2;

    ctx.beginPath();
    ctx.moveTo(p0.x, p0.y);
    ctx.lineTo(mxInitial, myInitial);
    ctx.strokeStyle = strokeColorFn(p0.opacity);
    ctx.lineWidth = lineWidthMultiplier * p0.opacity;
    ctx.stroke();

    // Draw quadratic curve segments between midpoints
    for (let i = 1; i < points.length - 1; i++) {
      const pt0 = points[i - 1];
      const pt1 = points[i];
      const pt2 = points[i + 1];

      const mx1 = (pt0.x + pt1.x) / 2;
      const my1 = (pt0.y + pt1.y) / 2;
      const mx2 = (pt1.x + pt2.x) / 2;
      const my2 = (pt1.y + pt2.y) / 2;

      const avgOpacity = (pt0.opacity + pt1.opacity + pt2.opacity) / 3;

      ctx.beginPath();
      ctx.moveTo(mx1, my1);
      ctx.quadraticCurveTo(pt1.x, pt1.y, mx2, my2);
      ctx.strokeStyle = strokeColorFn(avgOpacity);
      ctx.lineWidth = lineWidthMultiplier * avgOpacity;
      ctx.stroke();
    }

    // Draw last straight segment from last midpoint to the final point
    if (points.length >= 3) {
      const pLast2 = points[points.length - 2];
      const pLast = points[points.length - 1];
      const mxFinal = (pLast2.x + pLast.x) / 2;
      const myFinal = (pLast2.y + pLast.y) / 2;

      ctx.beginPath();
      ctx.moveTo(mxFinal, myFinal);
      ctx.lineTo(pLast.x, pLast.y);
      ctx.strokeStyle = strokeColorFn(pLast.opacity);
      ctx.lineWidth = lineWidthMultiplier * pLast.opacity;
      ctx.stroke();
    }
  }

  // Animation draw loop
  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Fade points slowly for a more persistent trail
    points.forEach((point) => {
      const fadeSpeed = isDrawing ? 0.015 : 0.03;
      point.opacity -= fadeSpeed;
    });

    // Remove faded out points
    points = points.filter(p => p.opacity > 0);

    if (points.length > 1) {
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      // Pass 1: Draw neon outer blue glow path
      ctx.shadowColor = 'rgba(0, 170, 255, 1)';
      ctx.shadowBlur = 16;
      drawCurve((opacity) => `rgba(0, 150, 255, ${opacity * 0.45})`, 10);

      // Pass 2: Draw thin sharp white inner core path
      ctx.shadowBlur = 4;
      drawCurve((opacity) => `rgba(255, 255, 255, ${opacity})`, 3.5);

      // Draw active tip glow dot
      if (isDrawing) {
        const tip = points[points.length - 1];
        ctx.beginPath();
        ctx.arc(tip.x, tip.y, 6, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = 'rgba(0, 170, 255, 1)';
        ctx.shadowBlur = 20;
        ctx.fill();
      }

      ctx.shadowBlur = 0; // reset
    }

    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
});
