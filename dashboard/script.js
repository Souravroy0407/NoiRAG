// ════════════════════════════════════════════════════════════════
//  NoiRAG Dashboard — Script
//  Noise-Aware Retrieval-Augmented Generation
// ════════════════════════════════════════════════════════════════

// ── Embedded Benchmark Data (from results/tables/*.json) ─────

const DATA = {
    gt: { "P@1": 0.6823, "R@5": 0.8736, "MRR": 0.7536, "NDCG@5": 0.7763, "F1@1": 0.6823 },
    semantic: {
        10: { noisy: { "P@1": 0.6723, "R@5": 0.8641, "MRR": 0.7443, "NDCG@5": 0.7666, "F1@1": 0.6723 } },
        25: {
            noisy:   { "P@1": 0.6359, "R@5": 0.8511, "MRR": 0.7188, "NDCG@5": 0.7437, "F1@1": 0.6359 },
            cleaned: { "P@1": 0.6568, "R@5": 0.8546, "MRR": 0.7323, "NDCG@5": 0.7567, "F1@1": 0.6568 }
        },
        50: { noisy: { "P@1": 0.5874, "R@5": 0.8227, "MRR": 0.6786, "NDCG@5": 0.7074, "F1@1": 0.5874 } },
        75: {
            noisy:   { "P@1": 0.5195, "R@5": 0.7717, "MRR": 0.6152, "NDCG@5": 0.6478, "F1@1": 0.5195 },
            cleaned: { "P@1": 1.0000, "R@5": 1.0000, "MRR": 1.0000, "NDCG@5": 0.9415, "F1@1": 1.0000 }
        }
    },
    formatting: {
        10: { noisy: { "P@1": 0.6738, "R@5": 0.8661, "MRR": 0.7482, "NDCG@5": 0.7701, "F1@1": 0.6738 } },
        25: {
            noisy:   { "P@1": 0.6678, "R@5": 0.8756, "MRR": 0.7477, "NDCG@5": 0.7729, "F1@1": 0.6678 },
            cleaned: { "P@1": 0.6543, "R@5": 0.8756, "MRR": 0.7377, "NDCG@5": 0.7640, "F1@1": 0.6543 }
        },
        50: { noisy: { "P@1": 0.6718, "R@5": 0.8751, "MRR": 0.7489, "NDCG@5": 0.7719, "F1@1": 0.6718 } },
        75: { noisy: { "P@1": 0.6718, "R@5": 0.8721, "MRR": 0.7496, "NDCG@5": 0.7732, "F1@1": 0.6718 } }
    }
};

// ── Text Comparison Samples ──────────────────────────────────

const SAMPLES = [
    {
        gt:      "Lift Yourself Up: Retrieval-augmented Text Generation with Self-Memory. Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu, Dongyan Zhao, Rui Yan. Peking University, Renmin University of China.",
        noisy:   "Lijft Youself Up: Retrieval-eugmented Text Generation wiht Self-Memory Xin Cheng, Di Lus, Xiuying Cben, Lemao Lqu, Dongyan Thao, Rui Yam. Peking Univresity, Renmin University of Ch.",
        cleaned: "Lift Yourself Up: Retrieval-augmented Text Generation with Self-Memory. In Chen, Di Us, Buying Chen, Lemma Liu, Donovan Zhao, Run An. Peking University, Renin University of China."
    },
    {
        gt:      "In-context learning (ICL) is a new paradigm for natural language processing that uses large language models to learn tasks given only a few examples. Unlike traditional fine-tuning approaches, ICL does not require updating the model parameters.",
        noisy:   "In-conxtet learnign (ICL) is a nwe prdaigm for natuarl lagnuage prcoessing taht uses lrage lagnuage mdoels to laern tsaks gvien olny a few exmaples. Unlkie traditoinal fnie-tuning approahces, ICL does nto reuqire updaitng the mdoel paramteres.",
        cleaned: "In-context learning (ICL) is a new paradigm for natural language processing that uses large language models to learn tasks given only a few examples. Unlike traditional fine-tuning approaches, ICL does not require updating the model parameters."
    },
    {
        gt:      "We propose a retrieval-augmented generation framework that leverages a dual-encoder architecture for efficient document retrieval from large-scale corpora. The framework employs FAISS indexing for sub-linear search complexity.",
        noisy:   "We prpoose a retrieval-augmetnde geneation framweork taht leveragse a dula-encoedr archtiecture for efifcient docuemnt retrieavl from lrage-scael corproa. The framweork emplosy FIASS indexnig for sub-lineaer seacrh complxeity.",
        cleaned: "We propose a retrieval-augmented generation framework that leverages a dual-encoder architecture for efficient document retrieval from large-scale corpora. The framework employs FAISS indexing for sub-linear search complexity."
    }
];

// ════════════════════════════════════════════════════════════════
//  CHART.JS CONFIGURATION
// ════════════════════════════════════════════════════════════════

const CHART_COLORS = {
    green:  { bg: 'rgba(16,185,129,0.7)',  border: '#10b981' },
    red:    { bg: 'rgba(239,68,68,0.7)',    border: '#ef4444' },
    blue:   { bg: 'rgba(59,130,246,0.7)',   border: '#3b82f6' },
    purple: { bg: 'rgba(139,92,246,0.5)',   border: '#8b5cf6' }
};

Chart.defaults.color = '#94a3b8';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.plugins.legend.labels.usePointStyle = true;

let barChart, radarChart;

function getBarData(noiseType) {
    const levels = [10, 25, 50, 75];
    const gtValues = levels.map(() => DATA.gt["P@1"]);
    const noisyValues = levels.map(l => DATA[noiseType][l].noisy["P@1"]);
    const cleanedValues = levels.map(l =>
        DATA[noiseType][l].cleaned ? DATA[noiseType][l].cleaned["P@1"] : null
    );

    return {
        labels: ['10%', '25%', '50%', '75%'],
        datasets: [
            {
                label: 'Ground Truth',
                data: gtValues,
                backgroundColor: CHART_COLORS.green.bg,
                borderColor: CHART_COLORS.green.border,
                borderWidth: 2, borderRadius: 6, barPercentage: 0.7
            },
            {
                label: 'Noisy Baseline',
                data: noisyValues,
                backgroundColor: CHART_COLORS.red.bg,
                borderColor: CHART_COLORS.red.border,
                borderWidth: 2, borderRadius: 6, barPercentage: 0.7
            },
            {
                label: 'NoiRAG Cleaned',
                data: cleanedValues,
                backgroundColor: CHART_COLORS.blue.bg,
                borderColor: CHART_COLORS.blue.border,
                borderWidth: 2, borderRadius: 6, barPercentage: 0.7
            }
        ]
    };
}

function getRadarData(noiseType) {
    const metrics = ["P@1", "R@5", "MRR", "NDCG@5", "F1@1"];
    const level = 75;
    const d = DATA[noiseType][level];

    return {
        labels: metrics,
        datasets: [
            {
                label: 'Ground Truth',
                data: metrics.map(m => DATA.gt[m]),
                borderColor: CHART_COLORS.green.border,
                backgroundColor: 'rgba(16,185,129,0.1)',
                pointBackgroundColor: CHART_COLORS.green.border,
                borderWidth: 2
            },
            {
                label: 'Noisy',
                data: metrics.map(m => d.noisy[m]),
                borderColor: CHART_COLORS.red.border,
                backgroundColor: 'rgba(239,68,68,0.1)',
                pointBackgroundColor: CHART_COLORS.red.border,
                borderWidth: 2
            },
            ...(d.cleaned ? [{
                label: 'NoiRAG Cleaned',
                data: metrics.map(m => d.cleaned[m]),
                borderColor: CHART_COLORS.blue.border,
                backgroundColor: 'rgba(59,130,246,0.1)',
                pointBackgroundColor: CHART_COLORS.blue.border,
                borderWidth: 2
            }] : [])
        ]
    };
}

function createCharts(noiseType) {
    const barCtx = document.getElementById('barChart').getContext('2d');
    const radarCtx = document.getElementById('radarChart').getContext('2d');

    if (barChart) barChart.destroy();
    if (radarChart) radarChart.destroy();

    barChart = new Chart(barCtx, {
        type: 'bar',
        data: getBarData(noiseType),
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { padding: 16, font: { size: 11 } } }
            },
            scales: {
                x: { grid: { display: false }, ticks: { font: { weight: 600 } } },
                y: {
                    beginAtZero: true, max: 1.05,
                    grid: { color: 'rgba(30,41,59,0.5)' },
                    ticks: { callback: v => (v * 100).toFixed(0) + '%' }
                }
            }
        }
    });

    radarChart = new Chart(radarCtx, {
        type: 'radar',
        data: getRadarData(noiseType),
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top', labels: { padding: 16, font: { size: 11 } } }
            },
            scales: {
                r: {
                    beginAtZero: true, max: 1.05,
                    grid: { color: 'rgba(30,41,59,0.4)' },
                    pointLabels: { font: { size: 11, weight: 600 } },
                    ticks: { display: false }
                }
            }
        }
    });
}

// ════════════════════════════════════════════════════════════════
//  TEXT COMPARISON
// ════════════════════════════════════════════════════════════════

function highlightDiffs(original, modified) {
    const origWords = original.split(/\s+/);
    const modWords = modified.split(/\s+/);
    let result = [];
    for (let i = 0; i < modWords.length; i++) {
        if (i < origWords.length && modWords[i] !== origWords[i]) {
            result.push(`<span class="error-word">${modWords[i]}</span>`);
        } else {
            result.push(modWords[i]);
        }
    }
    return result.join(' ');
}

function highlightFixes(original, cleaned) {
    const origWords = original.split(/\s+/);
    const cleanWords = cleaned.split(/\s+/);
    let result = [];
    for (let i = 0; i < cleanWords.length; i++) {
        if (i < origWords.length && cleanWords[i] !== origWords[i]) {
            result.push(`<span class="fixed-word">${cleanWords[i]}</span>`);
        } else {
            result.push(cleanWords[i]);
        }
    }
    return result.join(' ');
}

function showSample(index) {
    const s = SAMPLES[index];
    document.getElementById('gtText').textContent = s.gt;
    document.getElementById('noisyText').innerHTML = highlightDiffs(s.gt, s.noisy);
    document.getElementById('cleanedText').innerHTML = highlightFixes(s.noisy, s.cleaned);
}

// ════════════════════════════════════════════════════════════════
//  METRICS TABLE
// ════════════════════════════════════════════════════════════════

function buildMetricsTable() {
    const body = document.getElementById('metricsBody');
    const rows = [
        { label: 'Ground Truth (Clean)', cls: 'badge-gt', d: DATA.gt, recovery: null },
        { label: 'Semantic 10%',  cls: 'badge-noisy', d: DATA.semantic[10].noisy },
        { label: 'Semantic 25%',  cls: 'badge-noisy', d: DATA.semantic[25].noisy },
        { label: 'Semantic 50%',  cls: 'badge-noisy', d: DATA.semantic[50].noisy },
        { label: 'Semantic 75%',  cls: 'badge-noisy', d: DATA.semantic[75].noisy },
        { label: '↳ NoiRAG Cleaned (25%)', cls: 'badge-cleaned', d: DATA.semantic[25].cleaned, ref: DATA.semantic[25].noisy },
        { label: '↳ NoiRAG Cleaned (75%)', cls: 'badge-cleaned', d: DATA.semantic[75].cleaned, ref: DATA.semantic[75].noisy },
        { label: 'Formatting 10%', cls: 'badge-noisy', d: DATA.formatting[10].noisy },
        { label: 'Formatting 25%', cls: 'badge-noisy', d: DATA.formatting[25].noisy },
        { label: 'Formatting 50%', cls: 'badge-noisy', d: DATA.formatting[50].noisy },
        { label: 'Formatting 75%', cls: 'badge-noisy', d: DATA.formatting[75].noisy },
        { label: '↳ NoiRAG Cleaned (25%)', cls: 'badge-cleaned', d: DATA.formatting[25].cleaned, ref: DATA.formatting[25].noisy },
    ];

    rows.forEach(r => {
        const tr = document.createElement('tr');
        let recoveryHTML = '—';
        if (r.ref) {
            const delta = ((r.d["P@1"] - r.ref["P@1"]) * 100).toFixed(1);
            const cls = delta >= 0 ? 'recovery-positive' : 'recovery-negative';
            recoveryHTML = `<span class="recovery-badge ${cls}">${delta >= 0 ? '+' : ''}${delta}%</span>`;
        }
        tr.innerHTML = `
            <td class="label-cell ${r.cls}">${r.label}</td>
            <td>${(r.d["P@1"] * 100).toFixed(1)}%</td>
            <td>${(r.d["R@5"] * 100).toFixed(1)}%</td>
            <td>${(r.d["MRR"] * 100).toFixed(1)}%</td>
            <td>${(r.d["NDCG@5"] * 100).toFixed(1)}%</td>
            <td>${(r.d["F1@1"] * 100).toFixed(1)}%</td>
            <td>${recoveryHTML}</td>
        `;
        body.appendChild(tr);
    });
}

// ════════════════════════════════════════════════════════════════
//  INIT
// ════════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    createCharts('semantic');
    showSample(0);
    buildMetricsTable();

    document.getElementById('noiseSelector').addEventListener('change', e => {
        createCharts(e.target.value);
    });

    document.getElementById('sampleSelector').addEventListener('change', e => {
        showSample(parseInt(e.target.value));
    });
});
