// Update current year in footer
document.getElementById('current-year').textContent = new Date().getFullYear();

// Sample data for plots (replace with your actual data)
function createDailyViewingPlot() {
    const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    const avgDuration = [120, 95, 105, 115, 150, 180, 160];

    const trace = {
        x: days,
        y: avgDuration,
        type: 'bar',
        marker: {
            color: '#e50914'
        }
    };

    const layout = {
        title: 'Average Viewing Duration by Day',
        paper_bgcolor: '#181818',
        plot_bgcolor: '#181818',
        font: {
            color: '#ffffff'
        },
        xaxis: {
            gridcolor: '#282828'
        },
        yaxis: {
            gridcolor: '#282828',
            title: 'Duration (minutes)'
        }
    };

    Plotly.newPlot('dailyPlot', [trace], layout);
}

function createMonthlyTrendPlot() {
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const viewingTime = [140, 130, 125, 145, 160, 170, 155, 140, 150, 165, 175, 180];

    const trace = {
        x: months,
        y: viewingTime,
        type: 'scatter',
        mode: 'lines+markers',
        line: {
            color: '#e50914'
        }
    };

    const layout = {
        title: 'Monthly Viewing Trends',
        paper_bgcolor: '#181818',
        plot_bgcolor: '#181818',
        font: {
            color: '#ffffff'
        },
        xaxis: {
            gridcolor: '#282828'
        },
        yaxis: {
            gridcolor: '#282828',
            title: 'Average Duration (minutes)'
        }
    };

    Plotly.newPlot('monthlyPlot', [trace], layout);
}

function createBingeWatchingPlot() {
    const bingeData = {
        x: ['1-2 hrs', '2-3 hrs', '3-4 hrs', '4+ hrs'],
        y: [30, 45, 15, 10],
        type: 'bar',
        marker: {
            color: '#e50914'
        }
    };

    const layout = {
        title: 'Binge Watching Sessions Distribution',
        paper_bgcolor: '#181818',
        plot_bgcolor: '#181818',
        font: {
            color: '#ffffff'
        },
        xaxis: {
            gridcolor: '#282828',
            title: 'Session Duration'
        },
        yaxis: {
            gridcolor: '#282828',
            title: 'Frequency'
        }
    };

    Plotly.newPlot('bingePlot', [bingeData], layout);
}

function createMetricsPlot() {
    const metrics = ['Completion Rate', 'Session Consistency', 'Attention Score'];
    const values = [0.75, 0.82, 0.68];

    const trace = {
        x: metrics,
        y: values,
        type: 'bar',
        marker: {
            color: '#e50914'
        }
    };

    const layout = {
        title: 'Attention Span Metrics',
        paper_bgcolor: '#181818',
        plot_bgcolor: '#181818',
        font: {
            color: '#ffffff'
        },
        xaxis: {
            gridcolor: '#282828'
        },
        yaxis: {
            gridcolor: '#282828',
            title: 'Score',
            range: [0, 1]
        }
    };

    Plotly.newPlot('metricsPlot', [trace], layout);
}

// Create all plots when the page loads
document.addEventListener('DOMContentLoaded', () => {
    createDailyViewingPlot();
    createMonthlyTrendPlot();
    createBingeWatchingPlot();
    createMetricsPlot();

    // Populate findings
    const findings = [
        'Average viewing session duration: 125 minutes',
        'Most active viewing day: Saturday',
        'Binge-watching sessions increased by 25% over time',
        'Attention span shows consistent patterns on weekends',
        'Peak viewing hours: 8 PM - 11 PM'
    ];

    const findingsContainer = document.getElementById('findings-container');
    findings.forEach(finding => {
        const p = document.createElement('p');
        p.textContent = finding;
        findingsContainer.appendChild(p);
    });
});