document.addEventListener('DOMContentLoaded', () => {
  const controls = {
    support: document.getElementById('basket-support'),
    supportValue: document.getElementById('basket-support-value'),
    confidence: document.getElementById('basket-confidence'),
    confidenceValue: document.getElementById('basket-confidence-value'),
    category: document.getElementById('basket-category'),
    maxRules: document.getElementById('basket-max-rules'),
    sort: document.getElementById('basket-sort'),
    search: document.getElementById('basket-search')
  };
  const status = document.getElementById('basket-status');
  const tbody = document.getElementById('basket-tbody');
  const searchResults = document.getElementById('basket-search-results');
  const kpis = {
    orders: document.getElementById('basket-orders'),
    rules: document.getElementById('basket-rules-count'),
    basketSize: document.getElementById('basket-size')
  };

  let latestRules = [];

  const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }[char]));
  const pct = value => `${(Number(value || 0) * 100).toFixed(2)}%`;
  const num = value => Number(value || 0).toLocaleString('en-US', { maximumFractionDigits: 2 });

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
      min_support: controls.support.value,
      min_confidence: controls.confidence.value,
      category: controls.category.value,
      max_rules: controls.maxRules.value
    });
    return `/api/basket?${params.toString()}`;
  };

  const liftClass = lift => {
    if (Number(lift) > 2) return 'score-green';
    if (Number(lift) >= 1) return 'score-amber';
    return 'score-red';
  };

  const sortedRules = () => {
    const key = controls.sort.value;
    return [...latestRules].sort((a, b) => Number(b[key] || 0) - Number(a[key] || 0));
  };

  const renderKpis = data => {
    kpis.orders.textContent = num(data.summary?.total_orders);
    kpis.rules.textContent = num(data.summary?.total_rules);
    kpis.basketSize.textContent = num(data.summary?.avg_basket_size);
  };

  const renderTable = () => {
    const rules = sortedRules();
    if (!rules.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="table-empty">No association rules match the selected filters.</td></tr>';
      return;
    }
    tbody.innerHTML = rules.map(rule => `
      <tr>
        <td>${escapeHtml(rule.product_a)}</td>
        <td>${escapeHtml(rule.product_b)}</td>
        <td>${pct(rule.support)}</td>
        <td>${pct(rule.confidence)}</td>
        <td><span class="metric-pill ${liftClass(rule.lift)}">${num(rule.lift)}</span></td>
      </tr>
    `).join('');
  };

  const renderSearch = () => {
    const query = controls.search.value.trim().toLowerCase();
    if (!query) {
      searchResults.innerHTML = '<p class="table-empty">Search for a product to view cross-sell recommendations.</p>';
      return;
    }
    const matches = latestRules.filter(rule => (rule.antecedents || []).some(item => item.toLowerCase().includes(query)));
    if (!matches.length) {
      searchResults.innerHTML = '<p class="table-empty">No rules found with that product as the trigger.</p>';
      return;
    }
    searchResults.innerHTML = matches.map(rule => {
      const source = rule.antecedents.find(item => item.toLowerCase().includes(query)) || rule.product_a;
      return `
        <div class="recommendation-item">
          Customers who buy <strong>${escapeHtml(source)}</strong> also buy <strong>${escapeHtml(rule.product_b)}</strong>
          <span>(Confidence: ${pct(rule.confidence)}, Lift: ${num(rule.lift)})</span>
        </div>
      `;
    }).join('');
  };

  const loadBasket = async () => {
    controls.supportValue.textContent = Number(controls.support.value).toFixed(2);
    controls.confidenceValue.textContent = Number(controls.confidence.value).toFixed(2);
    setStatus('Mining basket rules...');
    try {
      const res = await fetch(buildUrl());
      const data = await res.json();
      if (!data.has_data) {
        window.location.reload();
        return;
      }
      latestRules = data.rules || [];
      syncSelectOptions(controls.category, data.filters?.categories || []);
      renderKpis(data);
      renderTable();
      renderSearch();
      setStatus('');
    } catch (error) {
      setStatus(`Basket analysis failed: ${error.message}`, true);
    }
  };

  [controls.support, controls.confidence].forEach(control => control.addEventListener('input', loadBasket));
  [controls.category, controls.maxRules].forEach(control => control.addEventListener('change', loadBasket));
  controls.sort.addEventListener('change', () => {
    renderTable();
  });
  controls.search.addEventListener('input', renderSearch);

  loadBasket();
});
