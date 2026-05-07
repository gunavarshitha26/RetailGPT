document.addEventListener('DOMContentLoaded', () => {
  const controls = {
    metric: document.getElementById('forecast-metric'),
    horizon: document.getElementById('forecast-horizon'),
    category: document.getElementById('forecast-category'),
    region: document.getElementById('forecast-region'),
    exportBtn: document.getElementById('forecast-export')
  };
  const status = document.getElementById('forecast-status');
  const kpis = {
    total: document.getElementById('forecast-total'),
    average: document.getElementById('forecast-average'),
    sum: document.getElementById('forecast-sum'),
    growth: document.getElementById('forecast-growth')
  };

  let forecastChart = null;
  let weeklyChart = null;
  let monthlyChart = null;
  let latestData = null;

  const fmt = (value, metric) => {
    const n = Number(value || 0);
    return '$' + n.toLocaleString('en-US', { maximumFractionDigits: 0 });
  };

  const destroyCharts = () => {
    [forecastChart, weeklyChart, monthlyChart].forEach(chart => chart && chart.destroy());
    forecastChart = null;
    weeklyChart = null;
    monthlyChart = null;
  };

  const setStatus = (message, isError = false) => {
    status.textContent = message || '';
    status.className = isError ? 'forecast-status error' : 'forecast-status';
  };

  const buildUrl = () => {
    const params = new URLSearchParams({
      metric: controls.metric.value,
      horizon: controls.horizon.value,
      category: controls.category.value,
      region: controls.region.value
    });
    return `/api/forecast?${params.toString()}`;
  };

  const calcKpis = (data) => {
    const metric = data.metric || controls.metric.value;
    const histValues = data.historical.map(p => Number(p.value || 0));
    const forecastValues = data.forecast.map(p => Number(p.value || 0));
    const total = histValues.reduce((a, b) => a + b, 0);
    const average = histValues.length ? total / histValues.length : 0;
    const forecastTotal = forecastValues.reduce((a, b) => a + b, 0);
    const prevPeriod = histValues.slice(-forecastValues.length).reduce((a, b) => a + b, 0);
    const growth = prevPeriod > 0 ? ((forecastTotal - prevPeriod) / prevPeriod) * 100 : 0;

    kpis.total.textContent = fmt(total, metric);
    kpis.average.textContent = fmt(average, metric);
    kpis.sum.textContent = fmt(forecastTotal, metric);
    kpis.growth.textContent = `${growth >= 0 ? '+' : ''}${growth.toFixed(1)}%`;
  };

  const renderForecastChart = (data) => {
    const ctx = document.getElementById('forecastChart');
    const hist = data.historical;
    const forecast = data.forecast;
    const labels = [...hist.map(p => p.date), ...forecast.map(p => p.date)];
    const histSeries = [...hist.map(p => p.value), ...forecast.map(() => null)];
    const forecastSeries = [...hist.map(() => null), ...forecast.map(p => p.value)];
    const lowerSeries = [...hist.map(() => null), ...forecast.map(p => p.lower)];
    const upperSeries = [...hist.map(() => null), ...forecast.map(p => p.upper)];

    forecastChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: 'Confidence Upper',
            data: upperSeries,
            borderColor: 'transparent',
            backgroundColor: 'rgba(59,130,246,0.14)',
            pointRadius: 0,
            fill: false
          },
          {
            label: 'Confidence Lower',
            data: lowerSeries,
            borderColor: 'transparent',
            backgroundColor: 'rgba(59,130,246,0.14)',
            pointRadius: 0,
            fill: '-1'
          },
          {
            label: 'Historical',
            data: histSeries,
            borderColor: 'rgba(148,163,184,0.8)',
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.35
          },
          {
            label: 'Forecast',
            data: forecastSeries,
            borderColor: '#3B82F6',
            borderDash: [6, 4],
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.35
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: '#94A3B8', filter: item => !item.text.includes('Confidence') } },
          tooltip: {
            backgroundColor: 'rgba(10,15,30,0.94)',
            callbacks: { label: ctx => `${ctx.dataset.label}: ${fmt(ctx.raw, data.metric)}` }
          }
        },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748B', maxTicksLimit: 12 } },
          y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748B' } }
        }
      }
    });
  };

  const renderSeasonality = (data) => {
    const metric = data.metric;
    const weekly = new Map(['Sun Mon Tue Wed Thu Fri Sat'.split(' ').map(day => [day, []])].flat());
    const monthly = new Map(['Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec'.split(' ').map(month => [month, []])].flat());

    data.historical.forEach(point => {
      const date = new Date(`${point.date}T00:00:00`);
      weekly.get(['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][date.getDay()]).push(point.value);
      monthly.get(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][date.getMonth()]).push(point.value);
    });

    const average = values => values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
    const barOptions = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: 'rgba(10,15,30,0.94)', callbacks: { label: ctx => fmt(ctx.raw, metric) } }
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#64748B' } },
        y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748B' } }
      }
    };

    weeklyChart = new Chart(document.getElementById('weeklyChart'), {
      type: 'bar',
      data: {
        labels: [...weekly.keys()],
        datasets: [{ data: [...weekly.values()].map(average), backgroundColor: '#3B82F6', borderRadius: 6 }]
      },
      options: barOptions
    });

    monthlyChart = new Chart(document.getElementById('monthlyChart'), {
      type: 'bar',
      data: {
        labels: [...monthly.keys()],
        datasets: [{ data: [...monthly.values()].map(average), backgroundColor: '#10B981', borderRadius: 6 }]
      },
      options: barOptions
    });
  };

  const loadForecast = async () => {
    destroyCharts();
    setStatus('Loading forecast...');
    try {
      const res = await fetch(buildUrl());
      const data = await res.json();
      latestData = data;

      if (!data.has_data) {
        window.location.reload();
        return;
      }
      if (!data.metric_available) {
        setStatus(data.message || 'Selected metric is not available in this dataset.', true);
        Object.values(kpis).forEach(el => { el.textContent = '--'; });
        return;
      }
      if (!data.historical?.length || !data.forecast?.length) {
        setStatus('No forecast data is available for the selected filters.', true);
        Object.values(kpis).forEach(el => { el.textContent = '--'; });
        return;
      }

      setStatus('');
      calcKpis(data);
      renderForecastChart(data);
      renderSeasonality(data);
    } catch (error) {
      setStatus(`Forecast failed: ${error.message}`, true);
    }
  };

  const exportCsv = () => {
    if (!latestData?.forecast?.length) return;
    const rows = [['date', 'forecast', 'lower', 'upper']];
    latestData.forecast.forEach(row => rows.push([row.date, row.value, row.lower, row.upper]));
    const csv = rows.map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `retailgpt_forecast_${latestData.metric}_${latestData.horizon}d.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  [controls.metric, controls.horizon, controls.category, controls.region].forEach(control => {
    control.addEventListener('change', loadForecast);
  });
  controls.exportBtn.addEventListener('click', exportCsv);
  loadForecast();
});
