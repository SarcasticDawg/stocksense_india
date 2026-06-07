let priceChart = null;

function renderPriceChart(ctx, labels, prices, period = '3mo', signals = []) {
    if (priceChart) {
        priceChart.destroy();
    }

    // Modern Gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, 'rgba(88, 166, 255, 0.3)');
    gradient.addColorStop(1, 'rgba(88, 166, 255, 0.0)');

    // Process Signal Markers
    const buyPoints = labels.map(l => {
        const sig = signals.find(s => s.date === l && s.verdict === 'BUY');
        return sig ? sig.price : null;
    });

    const sellPoints = labels.map(l => {
        const sig = signals.find(s => s.date === l && s.verdict === 'SELL');
        return sig ? sig.price : null;
    });

    priceChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Price',
                    data: prices,
                    borderColor: '#58a6ff',
                    backgroundColor: gradient,
                    fill: true,
                    borderWidth: 2,
                    tension: 0.2,
                    pointRadius: 0,
                    pointHoverRadius: 6,
                    zIndex: 1
                },
                {
                    label: 'BUY Signal',
                    data: buyPoints,
                    backgroundColor: '#238636',
                    borderColor: '#fff',
                    borderWidth: 1,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    showLine: false,
                    zIndex: 10
                },
                {
                    label: 'SELL Signal',
                    data: sellPoints,
                    backgroundColor: '#da3633',
                    borderColor: '#fff',
                    borderWidth: 1,
                    pointRadius: 6,
                    pointHoverRadius: 8,
                    showLine: false,
                    zIndex: 10
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#161b22',
                    titleColor: '#8b949e',
                    bodyColor: '#c9d1d9',
                    borderColor: '#30363d',
                    borderWidth: 1,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return `₹ ${context.parsed.y.toLocaleString()}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: { 
                        maxTicksLimit: 8, 
                        color: '#8b949e',
                        font: { size: 11 }
                    }
                },
                y: {
                    grid: { color: 'rgba(48, 54, 61, 0.5)' },
                    ticks: { 
                        color: '#8b949e',
                        font: { size: 11 },
                        callback: function(value) {
                            return '₹' + value;
                        }
                    }
                }
            }
        }
    });
}

async function updateChartPeriod(symbol, period) {
    // UI Feedback
    const buttons = document.querySelectorAll('.period-btn');
    buttons.forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[onclick*="${period}"]`).classList.add('active');

    try {
        const response = await fetch(`/api/chart/${symbol}?period=${period}`);
        const data = await response.json();
        
        if (data.labels && data.prices) {
            const ctx = document.getElementById('priceChart').getContext('2d');
            renderPriceChart(ctx, data.labels, data.prices, period);
        }
    } catch (error) {
        console.error('Error updating chart:', error);
    }
}
