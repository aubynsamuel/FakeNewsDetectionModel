document.addEventListener("DOMContentLoaded", () => {
  const darkModeToggle = document.getElementById("darkModeToggle");
  const currentTheme = localStorage.getItem("theme");

  if (currentTheme) {
    document.documentElement.setAttribute("data-theme", currentTheme);
    if (currentTheme === "dark") {
      darkModeToggle.checked = true;
    }
  }

  darkModeToggle.addEventListener("change", function () {
    if (this.checked) {
      document.documentElement.setAttribute("data-theme", "dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.setAttribute("data-theme", "light");
      localStorage.setItem("theme", "light");
    }
  });
});

async function analyzeHeadline() {
  const headline = document.getElementById("headline").value.trim();
  if (!headline) {
    alert("Please enter a headline.");
    return;
  }

  const loadingDiv = document.getElementById("loading");
  const resultsDiv = document.getElementById("results");
  const analyzeBtn = document.querySelector(".analyze-btn");
  const metricsExplanation = document.getElementById("metrics-explanation");

  loadingDiv.style.display = "block";
  resultsDiv.style.display = "none";
  metricsExplanation.style.display = "none";
  analyzeBtn.disabled = true;
  analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';

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
      displayError(data.error);
    } else {
      displayResults(data);
    }
  } catch (error) {
    displayError(
      "Could not connect to the analysis service. Please try again later."
    );
  } finally {
    loadingDiv.style.display = "none";
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<i class="fas fa-search"></i> Analyze Headline';
  }
}

function displayResults(data) {
  const resultsDiv = document.getElementById("results");
  const metricsExplanation = document.getElementById("metrics-explanation");
  resultsDiv.style.display = "block";
  metricsExplanation.style.display = "block";

  const { final_verdict } = data;
  const { verdict, confidence, score, components } = final_verdict;
  const details = data.components;

  const verdictConfig = {
    "Very High": {
      icon: "fas fa-check-circle",
      color: "#2ecc71",
    },
    High: {
      icon: "fas fa-check-circle",
      color: "#27ae60",
    },
    Moderate: {
      icon: "fas fa-exclamation-triangle",
      color: "#f39c12",
    },
    Low: {
      icon: "fas fa-times-circle",
      color: "#e67e22",
    },
    "Very Low": {
      icon: "fas fa-times-circle",
      color: "#e74c3c",
    },
  };

  const config = verdictConfig[confidence] || {
    icon: "fas fa-question-circle",
    color: "#95a5a6",
  };

  let metricsHTML = '<div class="metrics-grid">';
  for (const [key, value] of Object.entries(components)) {
    metricsHTML += `
          <div class="metric">
              <div class="metric-value" style="color: ${
                value >= 0.5 ? "#27ae60" : "#e74c3c"
              }">${value}</div>
              <div class="metric-label">${key.replace(/_/g, " ")}</div>
          </div>
      `;
  }
  metricsHTML += "</div>";

  // Details section
  let detailsHTML = "";
  if (details?.source_credibility) {
    detailsHTML += `<div class="metric-label" style="margin-top:10px;">Trusted Sources: <b>${details?.source_credibility?.trusted_count}</b> &nbsp; | &nbsp; Suspicious Sources: <b>${details?.source_credibility?.suspicious_count}</b> </div>`;
  }
  if (details?.network) {
    detailsHTML += `<div class="metric-label">Domain Diversity: <b>${details.network.domain_diversity}</b>
    </div>`;
  }

  resultsDiv.innerHTML = `
      <div class="verdict" style="background-color: ${
        config.color
      }20; border-left: 5px solid ${config.color};">
          <i class="${config.icon} verdict-icon" style="color: ${
    config.color
  };"></i>
          <div>
              <h2 style="color: ${
                config.color
              };" class="verdict-header">${verdict}</h2>
              <p> <strong>Score:</strong> ${score.toFixed(2)}/1.00</p>
          </div>
      </div>
      ${metricsHTML}
      <div class="metric-container">
        ${detailsHTML}
      </div>
      <div>
      </div>
      <div class="result-details">
          <p><strong>Headline Analyzed:</strong> ${data.headline}</p>
          <p><strong>Timestamp:</strong> ${new Date(
            data.timestamp
          ).toLocaleString()}</p>
      </div>
  `;
}

function displayError(message) {
  const resultsDiv = document.getElementById("results");
  resultsDiv.style.display = "block";
  resultsDiv.innerHTML = `
      <div class="verdict" style="background-color: #e74c3c20; border-left: 5px solid #e74c3c;">
          <i class="fas fa-exclamation-circle verdict-icon" style="color: #e74c3c;"></i>
          <div>
              <h2>Analysis Error</h2>
              <p>${message}</p>
          </div>
      </div>
  `;
}

// Allow Enter key to submit
document
  .getElementById("headline")
  .addEventListener("keydown", function (event) {
    if (event.key === "Enter" && event.ctrlKey) {
      analyzeHeadline();
    }
  });
