/**
 * Urban Acoustic Intelligence — Chart & Meter Visualizations (Chart.js)
 */

export class AcousticChartsManager {
  constructor(canvasElementId = 'sound-class-chart') {
    this.canvasId = canvasElementId;
    this.chart = null;

    // Palette mapped to sound categories
    this.palette = {
      air_conditioner: '#06b6d4',
      car_horn: '#f59e0b',
      children_playing: '#10b981',
      dog_bark: '#a855f7',
      drilling: '#ef4444',
      engine_idling: '#64748b',
      gun_shot: '#dc2626',
      jackhammer: '#f97316',
      siren: '#ec4899',
      street_music: '#3b82f6',
      ambient: '#14b8a6',
      unknown: '#475569',
    };

    this.initChart();
  }

  initChart() {
    const ctx = document.getElementById(this.canvasId);
    if (!ctx || typeof Chart === 'undefined') return;

    this.chart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: [],
        datasets: [
          {
            data: [],
            backgroundColor: [],
            borderWidth: 1,
            borderColor: '#0b0f19',
            hoverOffset: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'right',
            labels: {
              boxWidth: 10,
              padding: 8,
              color: '#94a3b8',
              font: {
                family: "'Inter', sans-serif",
                size: 10,
              },
            },
          },
          tooltip: {
            backgroundColor: 'rgba(11, 15, 25, 0.95)',
            borderColor: 'rgba(255, 255, 255, 0.1)',
            borderWidth: 1,
            titleColor: '#f8fafc',
            bodyColor: '#94a3b8',
            padding: 8,
            cornerRadius: 6,
          },
        },
        cutout: '68%',
      },
    });
  }

  /**
   * Update the sound category distribution chart.
   * @param {Object} classDistribution { sound_class_name: count }
   */
  updateClassDistribution(classDistribution) {
    if (!this.chart || !classDistribution) return;

    const entries = Object.entries(classDistribution)
      .filter(([_, count]) => count > 0)
      .sort((a, b) => b[1] - a[1]);

    const labels = entries.map(([name]) => name.replace('_', ' '));
    const data = entries.map(([_, count]) => count);
    const colors = entries.map(([name]) => this.palette[name] || '#6366f1');

    this.chart.data.labels = labels;
    this.chart.data.datasets[0].data = data;
    this.chart.data.datasets[0].backgroundColor = colors;
    this.chart.update();
  }

  /**
   * Update the visual live decibel meter.
   * @param {number} db Current decibel value
   */
  static updateDecibelMeter(db) {
    const fillEl = document.getElementById('meter-fill');
    const valueEl = document.getElementById('meter-db-value');
    if (!fillEl || !valueEl) return;

    const clampedDb = Math.max(30.0, Math.min(110.0, db));
    // Scale 30dB -> 110dB into 0% -> 100%
    const percentage = Math.round(((clampedDb - 30.0) / 80.0) * 100);

    fillEl.style.width = `${percentage}%`;
    valueEl.textContent = db.toFixed(1);

    // Color feedback
    if (db >= 85.0) {
      valueEl.style.color = '#ef4444';
    } else if (db >= 75.0) {
      valueEl.style.color = '#f97316';
    } else if (db >= 65.0) {
      valueEl.style.color = '#f59e0b';
    } else {
      valueEl.style.color = '#10b981';
    }
  }
}
