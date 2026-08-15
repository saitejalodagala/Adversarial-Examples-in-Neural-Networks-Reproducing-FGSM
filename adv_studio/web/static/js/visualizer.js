class StudioVisualizer {
  constructor() {
    this.landscapeChart = null;
    this.initLandscapeChart();
  }

  initLandscapeChart() {
    const canvas = document.getElementById('landscapeChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    this.landscapeChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Loss L(θ, x + γ·δ, y)',
          data: [],
          borderColor: '#3b82f6',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          fill: true,
          tension: 0.2,
          pointRadius: 2,
          pointHoverRadius: 6,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            title: { display: true, text: 'Interpolation Factor γ (0 = Clean, 1 = Adversarial)', color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' }
          },
          y: {
            title: { display: true, text: 'Cross-Entropy Loss', color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' }
          }
        },
        plugins: {
          legend: { labels: { color: '#f1f5f9' } },
          tooltip: {
            callbacks: {
              title: (items) => `γ = ${items[0].label}`,
              label: (context) => `Loss: ${context.parsed.y.toFixed(4)}`
            }
          }
        }
      }
    });
  }

  updateLandscape(data) {
    if (!this.landscapeChart) return;
    const labels = data.gammas.map(g => g.toFixed(2));
    this.landscapeChart.data.labels = labels;
    this.landscapeChart.data.datasets[0].data = data.losses;
    this.landscapeChart.update();
  }
}
