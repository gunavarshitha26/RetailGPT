document.addEventListener('DOMContentLoaded', () => {
  const controls = {
    metric: document.getElementById('anomaly-metric'),
    startDate: document.getElementById('anomaly-start-date'),
    endDate: document.getElementById('anomaly-end-date'),
    severity: document.getElementById('anomaly-severity'),
    category: document.getElementById('anomaly-category'),
    region: document.getElementById('anomaly-region'),
    exportBtn: document.getElementById('anomaly-export'),
    sortBtn: document.getElementById('sort-deviation')
  };
  const status = document.getElementById('anomaly-status');
  const tbody = document.getElementById('anomaly-tbody');
  const kpis = {
    total: document.getElementById('anomaly-total'),
    high: document.getElementById('anomaly-high'),
    medium: document.getElementById('anomaly-medium'),
    dateRange: document.getElementById('anomaly-date-range')
  };

  let latestData = null;
  let sortDescending = true;

  const money = value => '$' + Number(value || 0).toLocaleString('en-US', { maximumFractionDigits: 0 });
  const number = value => Number(value || 0).toLocaleString('en-US', { maximumFractionDigits: 2 });
  const metricFormat = (value, metric) => {
    return money(value);
  };

  const setStatus = (message, isError = false) => {
    status.textContent = message || '';
    status.className = isError ? 'forecast-status error' : 'forecast-status';
  };

  const syncSelectOptions = (select, values) => {
    const current = select.value || 'All';
    select.innerHTML = '<option>All</option>';
    values.forEach(value => {
      const option = document.createElement('option');
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    });
    select.value = [...select.options].some(option => option.value === current) ? current : 'All';
  };

  const buildUrl = () => {
    const params = new URLSearchParams({
      metric: controls.metric.value,
      severity: controls.severity.value,
      category: controls.category.value,
      region: controls.region.value
    });
    if (controls.startDate.value) params.set('start_date', controls.startDate.value);
    if (controls.endDate.value) params.set('end_date', controls.endDate.value);
    return `/api/anomalies?${params.toString()}`;
  };

  const renderKpis = data => {
    kpis.total.textContent = data.summary?.total ?? 0;
    kpis.high.textContent = data.summary?.high ?? 0;
    kpis.medium.textContent = data.summary?.medium ?? 0;
    kpis.dateRange.textContent = data.summary?.date_range || '--';
  };

  const renderTimeline = data => {
    const timeline = data.timeline || [];
    const container = document.getElementById('anomaly-timeline');
    if (!timeline.length) {
      container.innerHTML = '<p class="chart-empty">No timeline data for the selected filters.</p>';
      return;
    }

    const normal = timeline.filter(point => !point.is_anomaly);
    const anomalies = timeline.filter(point => point.is_anomaly);
    const lineColor = '#3B82F6';

    const traces = [
      {
        x: timeline.map(point => point.date),
        y: timeline.map(point => point.value),
        mode: 'lines',
        name: data.metric,
        line: { color: lineColor, width: 2 },
        hovertemplate: 'Date: %{x}<br>Value: %{y:,.2f}<extra></extra>'
      },
      {
        x: timeline.map(point => point.date),
        y: timeline.map(point => point.upper),
        mode: 'lines',
        name: 'Expected upper',
        line: { color: 'rgba(148,163,184,0.2)', width: 1 },
        hoverinfo: 'skip'
      },
      {
        x: timeline.map(point => point.date),
        y: timeline.map(point => point.lower),
        mode: 'lines',
        name: 'Expected range',
        fill: 'tonexty',
        fillcolor: 'rgba(148,163,184,0.12)',
        line: { color: 'rgba(148,163,184,0.2)', width: 1 },
        hoverinfo: 'skip'
      },
      {
        x: normal.map(point => point.date),
        y: normal.map(point => point.value),
        mode: 'markers',
        name: 'Normal',
        marker: { color: 'rgba(148,163,184,0.45)', size: 5 },
        hovertemplate: 'Date: %{x}<br>Value: %{y:,.2f}<extra></extra>'
      },
      {
        x: anomalies.map(point => point.date),
        y: anomalies.map(point => point.value),
        mode: 'markers',
        name: 'Anomaly',
        customdata: anomalies.map(point => [point.lower, point.upper, point.deviation_pct]),
        marker: { color: '#EF4444', size: 11, line: { color: 'rgba(255,255,255,0.7)', width: 1 } },
        hovertemplate: 'Date: %{x}<br>Value: %{y:,.2f}<br>Expected range: %{customdata[0]:,.2f} - %{customdata[1]:,.2f}<br>Deviation: %{customdata[2]:.1f}%<extra></extra>'
      }
    ];

    Plotly.newPlot(container, traces, {
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)',
      margin: { l: 56, r: 20, t: 8, b: 48 },
      font: { color: '#94A3B8', family: 'Inter, sans-serif' },
      legend: { orientation: 'h', y: 1.08 },
      xaxis: { gridcolor: 'rgba(255,255,255,0.05)' },
      yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' }
    }, { responsive: true, displayModeBar: false });
  };

  const severityBadge = severity => `<span class="severity-badge ${severity}">${severity}</span>`;

  const renderTable = records => {
    if (!records.length) {
      tbody.innerHTML = '<tr><td colspan="8" class="table-empty">No anomalies found for the selected filters.</td></tr>';
      return;
    }

    const sorted = [...records].sort((a, b) => {
      const result = Math.abs(b.deviation_pct) - Math.abs(a.deviation_pct);
      return sortDescending ? result : -result;
    });

    tbody.innerHTML = sorted.map(record => `
      <tr>
        <td>${record.date}</td>
        <td>${record.category}</td>
        <td>${record.region}</td>
        <td>${record.metric}</td>
        <td>${metricFormat(record.actual_value, record.metric)}</td>
        <td>${metricFormat(record.expected_value, record.metric)}</td>
        <td>${number(record.deviation_pct)}%</td>
        <td>${severityBadge(record.severity)}</td>
      </tr>
    `).join('');
  };

  const loadAnomalies = async () => {
    setStatus('Loading anomalies...');
    try {
      const res = await fetch(buildUrl());
      const data = await res.json();
      latestData = data;

      if (!data.has_data) {
        window.location.reload();
        return;
      }

      syncSelectOptions(controls.category, data.filters?.categories || []);
      syncSelectOptions(controls.region, data.filters?.regions || []);
      if (!controls.startDate.value) controls.startDate.value = data.filters?.min_date || '';
      if (!controls.endDate.value) controls.endDate.value = data.filters?.max_date || '';

      if (data.metric_available === false) {
        setStatus(data.message || 'Selected metric is not available in this dataset.', true);
        renderKpis(data);
        renderTimeline(data);
        renderTable([]);
        return;
      }

      setStatus('');
      renderKpis(data);
      renderTimeline(data);
      renderTable(data.anomalies || []);
    } catch (error) {
      setStatus(`Anomaly detection failed: ${error.message}`, true);
    }
  };

  const exportCsv = () => {
    const records = latestData?.anomalies || [];
    if (!records.length) return;

    const rows = [['Date', 'Category', 'Region', 'Metric', 'Actual', 'Expected', 'Deviation%', 'Severity']];
    records.forEach(record => {
      rows.push([
        record.date,
        record.category,
        record.region,
        record.metric,
        record.actual_value,
        record.expected_value,
        record.deviation_pct,
        record.severity
      ]);
    });

    const csv = rows.map(row => row.map(value => `"${String(value).replace(/"/g, '""')}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `retailgpt_anomalies_${controls.metric.value.toLowerCase()}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  [controls.metric, controls.startDate, controls.endDate, controls.severity, controls.category, controls.region].forEach(control => {
    control.addEventListener('change', loadAnomalies);
  });
  controls.exportBtn.addEventListener('click', exportCsv);
  controls.sortBtn.addEventListener('click', () => {
    sortDescending = !sortDescending;
    controls.sortBtn.textContent = sortDescending ? 'Sort by Deviation%' : 'Sort by Deviation% Asc';
    renderTable(latestData?.anomalies || []);
  });

  loadAnomalies();
});
