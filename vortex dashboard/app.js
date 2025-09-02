// Application data - from JSON input
const applicationData = {
  "sensor_locations": ["Building_A", "Building_B", "Factory_Floor", "Office_Area", "Warehouse"],
  "real_time_data": [
    {"location": "Building_A", "current_emission": 45.2, "predicted_emission": 47.8, "status": "Normal", "temperature": 22.5, "humidity": 48.2, "occupancy": 25, "timestamp": "2025-09-02 12:26"},
    {"location": "Building_B", "current_emission": 52.7, "predicted_emission": 51.3, "status": "Normal", "temperature": 21.8, "humidity": 52.1, "occupancy": 32, "timestamp": "2025-09-02 12:26"},
    {"location": "Factory_Floor", "current_emission": 142.8, "predicted_emission": 89.2, "status": "Critical", "temperature": 28.3, "humidity": 35.7, "occupancy": 45, "timestamp": "2025-09-02 12:26"},
    {"location": "Office_Area", "current_emission": 38.5, "predicted_emission": 41.2, "status": "Normal", "temperature": 23.1, "humidity": 45.8, "occupancy": 18, "timestamp": "2025-09-02 12:26"},
    {"location": "Warehouse", "current_emission": 67.3, "predicted_emission": 65.9, "status": "Warning", "temperature": 19.2, "humidity": 58.3, "occupancy": 8, "timestamp": "2025-09-02 12:26"}
  ],
  "model_performance": {"r2_score": 0.941, "mae": 5.27, "accuracy_percent": 94.1, "anomalies_detected": 216, "anomaly_rate": 5.0},
  "recommendations": [
    {"category": "Equipment Optimization", "priority": "High", "description": "Optimize equipment efficiency during high utilization periods", "potential_savings": 591.8, "status": "Pending"},
    {"category": "Anomaly Prevention", "priority": "Critical", "description": "Investigate frequent anomalies at Factory_Floor", "potential_savings": 847.2, "status": "In Progress"},
    {"category": "Energy Efficiency", "priority": "High", "description": "Improve energy efficiency with high emission-to-energy ratios", "potential_savings": 838.5, "status": "Pending"}
  ],
  "carbon_metrics": {"total_emissions_tons": 265.7, "daily_avg_kg": 1468, "monthly_avg_tons": 38.0, "target_reduction_percent": 35, "target_reduction_tons": 93.0, "offset_required_tons": 0.0, "potential_monthly_savings_tons": 2.28, "annual_cost_savings": 683.22},
  "feature_importance": [
    {"feature": "Energy Consumption", "importance": 96.1},
    {"feature": "Month", "importance": 0.6},
    {"feature": "Working Hours", "importance": 0.5},
    {"feature": "Humidity", "importance": 0.5}
  ]
};

class VortexDashboard {
  constructor() {
    this.currentData = [...applicationData.real_time_data];
    this.currentRecommendations = [...applicationData.recommendations];
    this.isRefreshing = false;
    this.filterLocation = '';
    this.init();
  }

  init() {
    this.setupEventListeners();
    this.updateDateTime();
    this.populateMonitoringTable();
    this.populateRecommendations();
    this.populateFeatureImportance();
    this.updateMetrics();
    
    // Update time every minute
    setInterval(() => this.updateDateTime(), 60000);
  }

  setupEventListeners() {
    // Refresh button
    document.getElementById('refresh-btn').addEventListener('click', () => {
      this.refreshData();
    });

    // Location filter
    document.getElementById('location-filter').addEventListener('change', (e) => {
      this.filterLocation = e.target.value;
      this.populateMonitoringTable();
      this.showNotification(`Filtered to: ${e.target.value || 'All Locations'}`, 'info');
    });

    // Export functionality
    document.getElementById('export-btn').addEventListener('click', () => {
      this.showExportModal();
    });

    document.getElementById('close-modal').addEventListener('click', () => {
      this.hideExportModal();
    });

    document.getElementById('cancel-export').addEventListener('click', () => {
      this.hideExportModal();
    });

    document.getElementById('confirm-export').addEventListener('click', () => {
      this.exportData();
    });

    // View trends button
    document.getElementById('view-trends').addEventListener('click', () => {
      this.showTrendAnalysis();
    });

    // Close modal on backdrop click
    document.getElementById('export-modal').addEventListener('click', (e) => {
      if (e.target.classList.contains('modal-overlay') || e.target.id === 'export-modal') {
        this.hideExportModal();
      }
    });

    // ESC key to close modal
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        this.hideExportModal();
      }
    });
  }

  updateDateTime() {
    const now = new Date();
    const options = {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
      timeZoneName: 'short'
    };
    const formattedDate = now.toLocaleDateString('en-US', options) + ' IST';
    document.getElementById('current-datetime').textContent = formattedDate;
  }

  populateMonitoringTable() {
    const tbody = document.getElementById('monitoring-tbody');
    const filteredData = this.filterLocation 
      ? this.currentData.filter(item => item.location === this.filterLocation)
      : this.currentData;

    tbody.innerHTML = filteredData.map(item => {
      const variance = ((item.current_emission - item.predicted_emission) / item.predicted_emission * 100).toFixed(1);
      const varianceClass = variance > 0 ? 'positive' : 'negative';
      const varianceSymbol = variance > 0 ? '+' : '';
      
      let actionButton = '';
      if (item.status === 'Critical') {
        actionButton = `<button class="action-btn acknowledge-btn" onclick="dashboard.acknowledgeAlert('${item.location}')">Acknowledge</button>`;
      } else if (item.status === 'Warning') {
        actionButton = `<button class="action-btn review-btn" onclick="dashboard.acknowledgeAlert('${item.location}')">Review</button>`;
      } else {
        actionButton = `<button class="action-btn details-btn" onclick="dashboard.viewDetails('${item.location}')">Details</button>`;
      }
      
      return `
        <tr>
          <td class="location-cell">${this.formatLocationName(item.location)}</td>
          <td>${item.current_emission.toFixed(1)}</td>
          <td>${item.predicted_emission.toFixed(1)}</td>
          <td class="variance ${varianceClass}">${varianceSymbol}${variance}%</td>
          <td><span class="status-badge ${item.status.toLowerCase()}">${item.status}</span></td>
          <td>${item.temperature.toFixed(1)}°C</td>
          <td>${actionButton}</td>
        </tr>
      `;
    }).join('');

    // Update last updated time
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    document.getElementById('last-updated').textContent = `Last updated: ${timeString}`;
  }

  populateRecommendations() {
    const container = document.getElementById('recommendations-list');
    
    container.innerHTML = this.currentRecommendations.map((rec, index) => `
      <div class="recommendation-item">
        <div class="recommendation-header">
          <span class="recommendation-category">${rec.category}</span>
          <span class="priority-badge ${rec.priority.toLowerCase()}">${rec.priority}</span>
        </div>
        <div class="recommendation-description">${rec.description}</div>
        <div class="recommendation-footer">
          <span class="savings-amount">💰 $${rec.potential_savings.toFixed(0)} savings</span>
          <div class="recommendation-actions">
            <span class="status-indicator ${rec.status.toLowerCase().replace(' ', '-')}">${rec.status}</span>
            ${rec.status !== 'Implemented' ? `
              <button class="btn btn--sm btn--primary implement-btn" onclick="dashboard.markAsImplemented(${index})">
                Mark Implemented
              </button>
            ` : ''}
          </div>
        </div>
      </div>
    `).join('');

    const activeCount = this.currentRecommendations.filter(rec => rec.status !== 'Implemented').length;
    document.getElementById('recommendations-count').textContent = 
      `${activeCount} active recommendations`;
  }

  populateFeatureImportance() {
    const container = document.getElementById('features-grid');
    
    container.innerHTML = applicationData.feature_importance.map(feature => `
      <div class="feature-item">
        <div class="feature-header">
          <span class="feature-name">${feature.feature}</span>
          <span class="feature-value">${feature.importance.toFixed(1)}%</span>
        </div>
        <div class="feature-bar">
          <div class="feature-fill" style="width: ${feature.importance}%"></div>
        </div>
      </div>
    `).join('');
  }

  updateMetrics() {
    // Calculate current emission rate (sum of all current emissions)
    const currentRate = this.currentData.reduce((sum, item) => sum + item.current_emission, 0);
    document.getElementById('current-rate').textContent = currentRate.toFixed(1);

    // Update daily total
    document.getElementById('daily-total').textContent = applicationData.carbon_metrics.daily_avg_kg.toLocaleString();

    // Calculate monthly progress
    const progress = Math.min(100, (applicationData.carbon_metrics.monthly_avg_tons / applicationData.carbon_metrics.target_reduction_tons) * 100);
    document.getElementById('monthly-progress').textContent = Math.round(progress);
    document.querySelector('.progress-fill').style.width = `${progress}%`;

    // Update active alerts
    const alerts = this.currentData.filter(item => item.status !== 'Normal');
    const criticalCount = alerts.filter(item => item.status === 'Critical').length;
    const warningCount = alerts.filter(item => item.status === 'Warning').length;
    
    document.getElementById('active-alerts').textContent = alerts.length;
    document.querySelector('.alert-breakdown').innerHTML = `
      <span class="alert-item critical">${criticalCount} Critical</span>
      <span class="alert-item warning">${warningCount} Warning</span>
    `;
  }

  refreshData() {
    if (this.isRefreshing) return;
    
    this.isRefreshing = true;
    const refreshBtn = document.getElementById('refresh-btn');
    refreshBtn.classList.add('refreshing');
    refreshBtn.innerHTML = '<span>🔄 Refreshing...</span>';

    // Simulate data refresh with slight variations
    setTimeout(() => {
      this.currentData = this.currentData.map(item => {
        const newEmission = item.current_emission + (Math.random() - 0.5) * 10;
        const newPredicted = item.predicted_emission + (Math.random() - 0.5) * 5;
        const newTemp = item.temperature + (Math.random() - 0.5) * 2;
        
        return {
          ...item,
          current_emission: Math.max(10, newEmission),
          predicted_emission: Math.max(10, newPredicted),
          temperature: Math.max(15, Math.min(35, newTemp)),
          status: this.determineStatus(newEmission, newPredicted),
          timestamp: new Date().toISOString().slice(0, 16).replace('T', ' ')
        };
      });

      this.populateMonitoringTable();
      this.updateMetrics();
      
      refreshBtn.classList.remove('refreshing');
      refreshBtn.innerHTML = '<span>🔄 Refresh Data</span>';
      this.isRefreshing = false;

      // Show success message
      this.showNotification('Data refreshed successfully', 'success');
    }, 2000);
  }

  determineStatus(current, predicted) {
    const variance = Math.abs(current - predicted) / predicted;
    if (variance > 0.3) return 'Critical';
    if (variance > 0.1) return 'Warning';
    return 'Normal';
  }

  acknowledgeAlert(location) {
    const item = this.currentData.find(d => d.location === location);
    if (!item) return;

    const originalStatus = item.status;
    
    if (item.status === 'Critical') {
      item.status = 'Warning';
      this.showNotification(`Critical alert for ${this.formatLocationName(location)} has been acknowledged and downgraded to Warning`, 'success');
    } else if (item.status === 'Warning') {
      item.status = 'Normal';
      this.showNotification(`Warning for ${this.formatLocationName(location)} has been resolved`, 'success');
    }

    this.populateMonitoringTable();
    this.updateMetrics();
  }

  viewDetails(location) {
    const item = this.currentData.find(d => d.location === location);
    if (!item) return;

    const details = `
      Location: ${this.formatLocationName(location)}
      Current Emission: ${item.current_emission.toFixed(1)} kg/h
      Predicted: ${item.predicted_emission.toFixed(1)} kg/h
      Temperature: ${item.temperature.toFixed(1)}°C
      Humidity: ${item.humidity.toFixed(1)}%
      Occupancy: ${item.occupancy} people
      Last Updated: ${item.timestamp}
    `;

    this.showNotification(`Details for ${this.formatLocationName(location)}:\n${details}`, 'info');
  }

  markAsImplemented(index) {
    if (index >= 0 && index < this.currentRecommendations.length) {
      const rec = this.currentRecommendations[index];
      rec.status = 'Implemented';
      
      this.showNotification(`Recommendation "${rec.category}" marked as implemented. Estimated savings: $${rec.potential_savings.toFixed(0)}`, 'success');
      this.populateRecommendations();
    }
  }

  showExportModal() {
    const modal = document.getElementById('export-modal');
    modal.classList.remove('hidden');
    // Focus first checkbox for accessibility
    setTimeout(() => {
      const firstCheckbox = modal.querySelector('input[type="checkbox"]');
      if (firstCheckbox) firstCheckbox.focus();
    }, 100);
  }

  hideExportModal() {
    document.getElementById('export-modal').classList.add('hidden');
  }

  exportData() {
    const checkedOptions = Array.from(document.querySelectorAll('.export-options input:checked'))
      .map(cb => cb.parentElement.textContent.trim());
    
    if (checkedOptions.length === 0) {
      this.showNotification('Please select at least one data type to export', 'error');
      return;
    }

    this.showNotification(`Preparing export: ${checkedOptions.join(', ')}`, 'info');
    
    // Simulate file preparation and download
    setTimeout(() => {
      const exportData = {
        timestamp: new Date().toISOString(),
        real_time_data: this.currentData,
        recommendations: this.currentRecommendations,
        carbon_metrics: applicationData.carbon_metrics,
        model_performance: applicationData.model_performance
      };

      const blob = new Blob([JSON.stringify(exportData, null, 2)], { 
        type: 'application/json' 
      });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `vortex_carbon_data_${new Date().toISOString().split('T')[0]}.json`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      
      this.showNotification('Data export completed successfully', 'success');
      this.hideExportModal();
    }, 1500);
  }

  showTrendAnalysis() {
    this.showNotification('Opening trend analysis dashboard...', 'info');
    
    // Simulate opening trend analysis in new window
    setTimeout(() => {
      const newWindow = window.open('about:blank', '_blank');
      if (newWindow) {
        newWindow.document.write(`
          <html>
            <head><title>Vortex Trend Analysis</title></head>
            <body style="font-family: Arial, sans-serif; padding: 20px; background: #1a1a1a; color: white;">
              <h1>🔬 Advanced Trend Analysis</h1>
              <p>This would open the full trend analysis dashboard with historical data visualization.</p>
              <p>Features would include:</p>
              <ul>
                <li>Historical emission trends</li>
                <li>Seasonal patterns</li>
                <li>Predictive forecasting</li>
                <li>Correlation analysis</li>
                <li>Optimization recommendations</li>
              </ul>
              <button onclick="window.close()" style="padding: 10px 20px; margin-top: 20px;">Close</button>
            </body>
          </html>
        `);
      } else {
        this.showNotification('Popup blocked. Please allow popups for this site.', 'error');
      }
    }, 500);
  }

  formatLocationName(location) {
    return location.replace('_', ' ');
  }

  showNotification(message, type = 'info') {
    // Remove existing notifications
    const existingNotifications = document.querySelectorAll('.notification');
    existingNotifications.forEach(n => n.remove());

    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification--${type}`;
    
    const typeColors = {
      success: 'var(--color-env-success)',
      error: 'var(--color-env-error)', 
      warning: 'var(--color-env-warning)',
      info: 'var(--color-env-primary)'
    };

    notification.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: var(--color-card-bg);
      border: 1px solid ${typeColors[type] || typeColors.info};
      border-radius: var(--radius-base);
      padding: var(--space-16);
      max-width: 400px;
      z-index: 2000;
      transform: translateX(100%);
      transition: transform 0.3s ease;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
      white-space: pre-line;
    `;

    const icons = {
      success: '✅',
      error: '❌',
      warning: '⚠️',
      info: 'ℹ️'
    };

    notification.innerHTML = `
      <div style="display: flex; align-items: flex-start; gap: var(--space-8);">
        <span style="font-size: var(--font-size-lg); flex-shrink: 0;">${icons[type] || icons.info}</span>
        <span style="color: var(--color-text); font-size: var(--font-size-sm); line-height: 1.4;">${message}</span>
      </div>
    `;

    document.body.appendChild(notification);

    // Animate in
    setTimeout(() => {
      notification.style.transform = 'translateX(0)';
    }, 100);

    // Auto remove after 5 seconds (longer for multi-line messages)
    const duration = message.length > 100 ? 7000 : 4000;
    setTimeout(() => {
      if (notification.parentNode) {
        notification.style.transform = 'translateX(100%)';
        setTimeout(() => {
          if (notification.parentNode) {
            document.body.removeChild(notification);
          }
        }, 300);
      }
    }, duration);
  }
}

// Initialize dashboard when page loads
let dashboard;
document.addEventListener('DOMContentLoaded', () => {
  dashboard = new VortexDashboard();

  // Add keyboard shortcuts
  addKeyboardShortcuts();
  
  // Add some visual enhancements
  addHoverEffects();
});

function addHoverEffects() {
  // Add hover effects to metric cards
  document.querySelectorAll('.metric-card').forEach(card => {
    card.addEventListener('mouseenter', () => {
      card.style.transform = 'translateY(-4px)';
    });
    
    card.addEventListener('mouseleave', () => {
      card.style.transform = 'translateY(0)';
    });
  });
}

function addKeyboardShortcuts() {
  document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + R for refresh (prevent default browser refresh)
    if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
      e.preventDefault();
      dashboard.refreshData();
    }
    
    // Ctrl/Cmd + E for export
    if ((e.ctrlKey || e.metaKey) && e.key === 'e') {
      e.preventDefault();
      dashboard.showExportModal();
    }
  });
}

// Simulate real-time updates every 45 seconds
setInterval(() => {
  if (!dashboard.isRefreshing) {
    // Subtle data updates to simulate real-time monitoring
    dashboard.currentData.forEach(item => {
      // Small random variations
      const emissionChange = (Math.random() - 0.5) * 3;
      const tempChange = (Math.random() - 0.5) * 0.8;
      
      item.current_emission += emissionChange;
      item.temperature += tempChange;
      
      // Keep values within realistic bounds
      item.current_emission = Math.max(15, Math.min(200, item.current_emission));
      item.temperature = Math.max(15, Math.min(35, item.temperature));
      
      // Update status based on new values
      item.status = dashboard.determineStatus(item.current_emission, item.predicted_emission);
    });
    
    dashboard.populateMonitoringTable();
    dashboard.updateMetrics();
  }
}, 45000);

// Add pulse animation to critical alerts every 3 seconds
setInterval(() => {
  const criticalBadges = document.querySelectorAll('.status-badge.critical');
  criticalBadges.forEach(badge => {
    badge.style.animation = 'none';
    setTimeout(() => {
      badge.style.animation = 'pulse 2s infinite';
    }, 10);
  });
}, 3000);

// Performance monitoring
if ('performance' in window) {
  window.addEventListener('load', () => {
    setTimeout(() => {
      const perfData = performance.getEntriesByType('navigation')[0];
      console.log(`Vortex Dashboard loaded in ${Math.round(perfData.loadEventEnd - perfData.fetchStart)}ms`);
    }, 0);
  });
}