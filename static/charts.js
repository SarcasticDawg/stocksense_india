let priceChart = null;
let rsiChart = null;
let currentChartMode = 'line';

function calculateRSI(closes, period = 14) {
    let rsi = new Array(closes.length).fill(null);
    if (closes.length <= period) return rsi;
    
    let gains = 0, losses = 0;
    for (let i = 1; i <= period; i++) {
        let diff = closes[i] - closes[i - 1];
        if (diff >= 0) gains += diff;
        else losses -= diff;
    }
    let avgGain = gains / period;
    let avgLoss = losses / period;
    
    rsi[period] = 100 - (100 / (1 + avgGain / (avgLoss === 0 ? 1e-10 : avgLoss)));
    
    for (let i = period + 1; i < closes.length; i++) {
        let diff = closes[i] - closes[i - 1];
        let gain = diff >= 0 ? diff : 0;
        let loss = diff < 0 ? -diff : 0;
        
        avgGain = (avgGain * (period - 1) + gain) / period;
        avgLoss = (avgLoss * (period - 1) + loss) / period;
        
        rsi[i] = 100 - (100 / (1 + avgGain / (avgLoss === 0 ? 1e-10 : avgLoss)));
    }
    return rsi;
}

const crosshairPlugin = {
    id: 'crosshairSync',
    afterDraw: (chart) => {
        if (chart.crosshairX) {
            const ctx = chart.ctx;
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(chart.crosshairX, chart.chartArea.top);
            ctx.lineTo(chart.crosshairX, chart.chartArea.bottom);
            ctx.lineWidth = 1;
            ctx.strokeStyle = 'rgba(108, 117, 125, 0.5)';
            ctx.setLineDash([5, 5]);
            ctx.stroke();
            ctx.restore();
        }
    }
};

const handleHover = (e, sourceChart, targetChart) => {
    if (!targetChart || !sourceChart || !sourceChart.ctx) return;
    const points = sourceChart.getElementsAtEventForMode(e, 'index', {intersect: false}, true);
    
    if (points.length) {
        const index = points[0].index;
        const targetMeta = targetChart.getDatasetMeta(0);
        
        if(targetMeta && targetMeta.data[index]) {
             const targetPoint = {datasetIndex: 0, index: index};
             if (targetChart.tooltip) {
                 targetChart.tooltip.setActiveElements([targetPoint], {x: targetMeta.data[index].x, y: targetMeta.data[index].y});
             }
             targetChart.crosshairX = targetMeta.data[index].x;
             sourceChart.crosshairX = points[0].element.x;
             targetChart.update('none');
             sourceChart.update('none');
        }
    } else {
        if (targetChart.tooltip) {
            targetChart.tooltip.setActiveElements([], {x: 0, y: 0});
        }
        sourceChart.crosshairX = null;
        targetChart.crosshairX = null;
        targetChart.update('none');
        sourceChart.update('none');
    }
};

function renderRsiChart(dataObj) {
    if (rsiChart) rsiChart.destroy();
    const ctx = document.getElementById('rsiChart').getContext('2d');
    
    // Parse dates to luxon objects for reliable time axis
    const timeData = dataObj.labels.map(l => luxon.DateTime.fromISO(l).toMillis());
    const rsiRaw = calculateRSI(dataObj.prices);
    const rsiPoints = timeData.map((t, i) => ({x: t, y: rsiRaw[i]}));

    rsiChart = new Chart(ctx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'RSI',
                data: rsiPoints,
                borderWidth: 1.5,
                pointRadius: 0,
                pointHoverRadius: 0,
                fill: false,
                borderColor: '#EF9F27'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            onHover: (e) => handleHover(e, rsiChart, priceChart),
            plugins: {
                legend: { display: false },
                annotation: {
                    annotations: {
                        boxHigh: { type: 'box', yMin: 70, yMax: 100, backgroundColor: 'rgba(226,75,74,0.12)', borderWidth: 0 },
                        boxLow: { type: 'box', yMin: 0, yMax: 30, backgroundColor: 'rgba(99,153,34,0.12)', borderWidth: 0 },
                        line70: { type: 'line', yMin: 70, yMax: 70, borderColor: '#E24B4A', borderDash: [5,5], borderWidth: 1 },
                        line50: { type: 'line', yMin: 50, yMax: 50, borderColor: '#8b949e', borderDash: [5,5], borderWidth: 1 },
                        line30: { type: 'line', yMin: 30, yMax: 30, borderColor: '#639922', borderDash: [5,5], borderWidth: 1 }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'day', tooltipFormat: 'MMM d, yyyy' },
                    grid: { display: false },
                    ticks: { maxTicksLimit: 8, color: '#6c757d', font: { size: 11 } }
                },
                y: {
                    min: 0, max: 100,
                    grid: { color: '#f1f3f5' },
                    ticks: {
                        callback: function(value) { return [30, 50, 70].includes(value) ? value : null; },
                        color: '#6c757d', font: { size: 10 }
                    }
                }
            }
        },
        plugins: [crosshairPlugin]
    });
}

function renderPriceChart(dataObj, mode) {
    if (priceChart) priceChart.destroy();
    const ctx = document.getElementById('priceChart').getContext('2d');

    const timeData = dataObj.labels.map(l => luxon.DateTime.fromISO(l).toMillis());

    const buyPoints = dataObj.labels.map((l, i) => {
        const sig = dataObj.signals.find(s => s.date === l && s.verdict === 'BUY');
        return sig ? {x: timeData[i], y: sig.price} : null;
    }).filter(p => p !== null);

    const sellPoints = dataObj.labels.map((l, i) => {
        const sig = dataObj.signals.find(s => s.date === l && s.verdict === 'SELL');
        return sig ? {x: timeData[i], y: sig.price} : null;
    }).filter(p => p !== null);

    let datasets = [];
    let yMin = null;
    let yMax = null;

    if (mode === 'candle') {
        const candleData = [];
        dataObj.ohlc.forEach((d, i) => {
            if (d.o != null && d.h != null && d.l != null && d.c != null) {
                candleData.push({ x: timeData[i], o: d.o, h: d.h, l: d.l, c: d.c });
                if (yMin === null || d.l < yMin) yMin = d.l;
                if (yMax === null || d.h > yMax) yMax = d.h;
            }
        });

        datasets.push({
            label: 'Price',
            type: 'candlestick',
            data: candleData,
            color: { up: '#26a69a', down: '#ef5350', unchanged: '#999' },
            borderColor: { up: '#26a69a', down: '#ef5350', unchanged: '#999' },
            borderWidth: 1
        });
    } else {
        const lineData = [];
        dataObj.prices.forEach((p, i) => {
            if (p != null) {
                lineData.push({x: timeData[i], y: p});
                if (yMin === null || p < yMin) yMin = p;
                if (yMax === null || p > yMax) yMax = p;
            }
        });

        datasets.push({
            label: 'Price',
            type: 'line',
            data: lineData,
            borderColor: '#007bff',
            backgroundColor: 'rgba(0, 123, 255, 0.1)',
            fill: true,
            tension: 0.3,
            pointRadius: 0
        });
    }

    datasets.push({
        label: 'BUY Signal', type: 'scatter', data: buyPoints,
        backgroundColor: '#28a745', borderColor: '#fff', borderWidth: 1, pointRadius: 6, zIndex: 10
    });
    datasets.push({
        label: 'SELL Signal', type: 'scatter', data: sellPoints,
        backgroundColor: '#dc3545', borderColor: '#fff', borderWidth: 1, pointRadius: 6, zIndex: 10
    });

    if (yMin !== null) {
        yMin *= 0.98;
        yMax *= 1.02;
    }

    priceChart = new Chart(ctx, {
        type: mode === 'candle' ? 'candlestick' : 'line',
        data: { datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { intersect: false, mode: 'index' },
            onHover: (e) => handleHover(e, priceChart, rsiChart),
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            if (context.dataset.type === 'candlestick') {
                                return `O: ₹${context.raw.o} H: ₹${context.raw.h} L: ₹${context.raw.l} C: ₹${context.raw.c}`;
                            }
                            return `₹ ${context.parsed.y}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'time',
                    time: { unit: 'day' },
                    grid: { display: false },
                    ticks: { display: false }
                },
                y: {
                    min: yMin || undefined,
                    max: yMax || undefined,
                    grid: { color: '#f1f3f5' },
                    ticks: { callback: function(value) { return '₹' + value; } }
                }
            }
        },
        plugins: [crosshairPlugin]
    });
}

function setChartMode(mode) {
    try {
        currentChartMode = mode;
        const btnLine = document.getElementById('btn-line');
        const btnCandle = document.getElementById('btn-candle');
        if (btnLine) btnLine.classList.toggle('active', mode === 'line');
        if (btnCandle) btnCandle.classList.toggle('active', mode === 'candle');
        
        const rsiContainer = document.getElementById('rsi-chart-container');
        const priceContainer = document.getElementById('price-chart-container');
        
        if (mode === 'line') {
            if (rsiContainer) rsiContainer.style.display = 'none';
            if (priceContainer) priceContainer.style.height = '450px';
        } else {
            if (rsiContainer) rsiContainer.style.display = 'block';
            if (priceContainer) priceContainer.style.height = '300px';
        }
        
        if (window.chartData) {
            renderPriceChart(window.chartData, mode);
            if (mode === 'candle') renderRsiChart(window.chartData);
        }
    } catch (e) {
        console.error("setChartMode failed:", e);
    }
}

async function updateChartPeriod(symbol, period) {
    try {
        const response = await fetch(`/api/chart/${symbol}?period=${period}`);
        const data = await response.json();
        if (data.labels) {
            window.chartData = data;
            window.chartData.signals = window.chartData.signals || [];
            setChartMode(currentChartMode);
        }
    } catch (error) {
        console.error(error);
    }
}
