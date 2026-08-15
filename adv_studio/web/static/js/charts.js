class StudioCharts {
  constructor() {
    this.probChart = null;
    this.initProbabilityChart();
  }

  initProbabilityChart() {
    const ctx = document.getElementById('probChart').getContext('2d');
    const labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'];

    this.probChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Clean Input',
            data: new Array(10).fill(0),
            backgroundColor: 'rgba(16, 185, 129, 0.7)',
            borderColor: '#10b981',
            borderWidth: 1,
            borderRadius: 4,
          },
          {
            label: 'Adversarial Input',
            data: new Array(10).fill(0),
            backgroundColor: 'rgba(239, 68, 68, 0.7)',
            borderColor: '#ef4444',
            borderWidth: 1,
            borderRadius: 4,
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 300 },
        scales: {
          y: {
            beginAtZero: true,
            max: 1.0,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: {
              color: '#94a3b8',
              callback: (value) => (value * 100).toFixed(0) + '%'
            }
          },
          x: {
            grid: { display: false },
            ticks: { color: '#f1f5f9', font: { weight: 'bold' } }
          }
        },
        plugins: {
          legend: {
            position: 'top',
            labels: { color: '#f1f5f9', font: { size: 11 } }
          },
          tooltip: {
            callbacks: {
              label: (context) => `${context.dataset.label}: ${(context.parsed.y * 100).toFixed(2)}%`
            }
          }
        }
      }
    });
  }

  updateProbabilities(cleanProbs, advProbs = null) {
    if (!this.probChart) return;
    this.probChart.data.datasets[0].data = cleanProbs;
    if (advProbs) {
      this.probChart.data.datasets[1].data = advProbs;
      this.probChart.data.datasets[1].hidden = false;
    } else {
      this.probChart.data.datasets[1].data = new Array(10).fill(0);
      this.probChart.data.datasets[1].hidden = true;
    }
    this.probChart.update();
  }
}
