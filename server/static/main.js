async function analyzeHeadline() {
  const headline = document.getElementById("headline").value.trim();
  const loadingDiv = document.getElementById("loading");
  const resultsDiv = document.getElementById("results");
  const analyzeBtn = document.querySelector(".analyze-btn");

  if (!headline) {
    alert("Please enter a headline to analyze");
    return;
  }

  if (headline.length < 10) {
    alert("Please enter a more complete headline (at least 10 characters)");
    return;
  }

  // Show loading state
  loadingDiv.style.display = "block";
  resultsDiv.style.display = "none";
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "🔄 Analyzing...";

  try {
    const response = await fetch("/analyze", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ headline: headline }),
    });

    const data = await response.json();

    // test data
    // const response = await fetch("/static/serverResponseFormat.json");
    // const data = await response.json();

    if (data.error) {
      showError(data.error);
    } else {
      showResults(data);
    }
  } catch (error) {
    showError(error);
    // showError("Network error. Please check your connection and try again.");
  } finally {
    // Hide loading state
    loadingDiv.style.display = "none";
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "🚀 Analyze Headline";
  }
}

function showResults(data) {
  const resultsDiv = document.getElementById("results");

  // Map confidence to color and icon
  const verdictIcons = {
    "Very High": "✅",
    High: "🟢",
    Moderate: "⚠️",
    Low: "🟠",
    "Very Low": "🚫",
  };
  const verdictColors = {
    "Very High": "#48bb78",
    High: "#38a169",
    Moderate: "#ed8936",
    Low: "#ed8936",
    "Very Low": "#f56565",
  };

  const verdict = data.final_verdict?.verdict || "N/A";
  const confidence = data.final_verdict?.confidence || "N/A";
  const score =
    data.final_verdict?.score !== undefined ? data.final_verdict.score : "N/A";
  const components = data.final_verdict?.components || {};
  const details = data.components || {};

  // Build metrics table from components
  function metricRow(label, value, color) {
    return `<div class="metric">
            <div class="metric-value" style="color: ${color}">${value}</div>
            <div class="metric-label">${label}</div>
        </div>`;
  }

  let metricsHTML = "";
  if (Object.keys(components).length > 0) {
    metricsHTML = `<div class="metrics">
            ${metricRow(
              "Claim Verification",
              components.claim_verification,
              components.claim_verification >= 0.6 ? "#48bb78" : "#f56565"
            )}
            ${metricRow(
              "Source Credibility",
              components.source_credibility,
              components.source_credibility >= 0.6 ? "#48bb78" : "#f56565"
            )}
            ${metricRow(
              "Clickbait Detection",
              components.clickbait_detection,
              components.clickbait_detection >= 0.6 ? "#48bb78" : "#f56565"
            )}
            ${metricRow(
              "Network Propagation",
              components.network_propagation,
              components.network_propagation >= 0.6 ? "#48bb78" : "#f56565"
            )}
        </div>`;
  }

  // Details section
  let detailsHTML = "";
  if (details.source_credibility) {
    detailsHTML += `<div class="metric-label" style="margin-top:10px;">Trusted Sources: <b>${details.source_credibility.trusted_count}</b> &nbsp; | &nbsp; Suspicious Sources: <b>${details.source_credibility.suspicious_count}</b></div>`;
  }
  if (details.network) {
    detailsHTML += `<div class="metric-label">Domain Diversity: <b>${details.network.domain_diversity}</b></div>`;
  }

  resultsDiv.className = "results";
  resultsDiv.innerHTML = `
        <div class="verdict">
            <div class="verdict-icon">${verdictIcons[confidence] || "❓"}</div>
            <div class="verdict-text">
                <h2 style="color: ${
                  verdictColors[confidence] || "#718096"
                }">${verdict}</h2>
                <p>Confidence: ${confidence} • Score: ${score}/1.00</p>
            </div>
        </div>
        ${metricsHTML}
        ${detailsHTML}
        <div class="metric-label" style="margin-top:10px;">Headline: <b>${
          data.headline || ""
        }</b></div>
        <div class="metric-label">Timestamp: <b>${
          data.timestamp || ""
        }</b></div>
    `;
  resultsDiv.style.display = "block";
}

function showError(message) {
  const resultsDiv = document.getElementById("results");
  resultsDiv.className = "results";
  resultsDiv.innerHTML = `
        <div class="error">
            <strong>❌ Error:</strong> ${message}
        </div>
    `;
  resultsDiv.style.display = "block";
}

// Allow Enter key to submit
document
  .getElementById("headline")
  .addEventListener("keydown", function (event) {
    if (event.key === "Enter" && event.ctrlKey) {
      analyzeHeadline();
    }
  });
