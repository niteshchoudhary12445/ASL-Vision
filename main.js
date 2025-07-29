const classes = JSON.parse(document.getElementById('class-data').textContent);
const ctx = document.getElementById('probChart').getContext('2d');
const probChart = new Chart(ctx, {
  type: 'bar',
  data: { labels: classes, datasets: [{ label:'Confidence', data: Array(classes.length).fill(0) }] },
  options: {
    responsive: true,
    animation: { duration: 0 },
    scales: {
      y: { beginAtZero: true, max: 1 },
      x: { ticks: { color: getComputedStyle(document.body).color } },
      y: { ticks: { color: getComputedStyle(document.body).color } }
    }
  }
});

const videoImg = document.getElementById('video');
const spinner = document.getElementById('spinner');
const historyList = document.getElementById('historyList');
const sentenceBuilder = document.getElementById('sentenceBuilder');
const speakBtn = document.getElementById('speakBtn');
const clearBtn = document.getElementById('clearBtn');
const slider = document.getElementById('sensitivity');
const sensVal = document.getElementById('sensVal');
const themeToggle = document.getElementById('themeToggle');
const backspaceBtn = document.getElementById('backspaceBtn');
let lastLabel = '';

function handleStreamError() {
  spinner.style.display = 'none';
  console.warn('Stream failed, retrying...');
  setTimeout(() => {
    spinner.style.display = 'block';
    videoImg.src = '/video_feed?ts=' + Date.now();
  }, 1000);
}

async function fetchData() {
  try {
    const res = await fetch('/probabilities');
    const { probs, history } = await res.json();
    probChart.data.datasets[0].data = probs;
    probChart.update('none');
    historyList.innerHTML = '';
    history.forEach(l => {
      const li = document.createElement('li');
      li.className = 'list-group-item';
      li.textContent = l;
      historyList.appendChild(li);
    });
    const newL = history[0];
    if (newL && newL !== 'No hand' && newL !== lastLabel) {
      sentenceBuilder.textContent += newL;
      lastLabel = newL;
    }
  } catch(e) {
    console.error(e);
  }
}

videoImg.onerror = handleStreamError;
setInterval(fetchData, 200);
fetchData();

slider.addEventListener('input', () => {
  const v = slider.value;
  sensVal.textContent = v;
  fetch('/set_sensitivity', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ value: v })
  }).catch(console.error);
});

speakBtn.addEventListener('click', () => {
  const t = sentenceBuilder.textContent.trim();
  if (t) window.speechSynthesis.speak(new SpeechSynthesisUtterance(t));
});

backspaceBtn.addEventListener('click', () => {
  let text = sentenceBuilder.textContent;
  // Remove last character (could be a space or letter)
  if (text.length > 0) {
    sentenceBuilder.textContent = text.slice(0, -1);
  }
});

clearBtn.addEventListener('click', () => {
  sentenceBuilder.textContent = '';
  lastLabel = '';
});

themeToggle.addEventListener('click', () => {
  document.body.classList.toggle('dark-theme');
  document.body.classList.toggle('light-theme');
  themeToggle.textContent = document.body.classList.contains('dark-theme') ? '☀️' : '🌙';
  // Update chart tick colors
  probChart.options.scales.x.ticks.color = getComputedStyle(document.body).color;
  probChart.options.scales.y.ticks.color = getComputedStyle(document.body).color;
  probChart.update();
});

const suggestionBox = document.getElementById('suggestionBox');

function getCurrentPrefix(text) {
  return text.split(/[\s.,!?]/).pop();
}

function fetchSuggestions(prefix) {
  if (!prefix) {
    suggestionBox.innerHTML = '';
    return;
  }

  fetch(`/suggest?prefix=${encodeURIComponent(prefix)}`)
    .then(res => res.json())
    .then(data => {
      suggestionBox.innerHTML = '';
      data.suggestions.forEach(word => {
        const btn = document.createElement('button');
        btn.className = 'btn btn-sm btn-outline-primary m-1';
        btn.textContent = word;
        btn.onclick = () => {
          const base = sentenceBuilder.textContent.trimEnd();
          const words = base.split(/\s+/);
          words.pop();  // remove current prefix
          words.push(word);  // add full word
          sentenceBuilder.textContent = words.join(' ') + ' ';
        };
        suggestionBox.appendChild(btn);
      });
    })
    .catch(console.error);
}

// Monitor for character updates and request suggestions
setInterval(() => {
  const text = sentenceBuilder.textContent.trim();
  const prefix = getCurrentPrefix(text);
  fetchSuggestions(prefix);
}, 500);

