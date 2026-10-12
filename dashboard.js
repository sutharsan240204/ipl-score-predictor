const API_BASE = ""; // If your API is hosted elsewhere, set its base URL here.
let scoreComparisonChart;

const $ = (id) => document.getElementById(id);
const setText = (id, value) => { $(id).textContent = value ?? "—"; };

async function getJSON(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options
  });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try { const body = await response.json(); detail = body.error || body.message || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response.json();
}

function fillSelect(id, values, placeholder) {
  const select = $(id);
  select.innerHTML = `<option value="">${placeholder}</option>`;
  (values || []).forEach(value => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.appendChild(option);
  });
}

async function loadOptions() {
  try {
    const data = await getJSON("/api/options");
    fillSelect("battingTeam", data.teams, "Select batting team");
    fillSelect("bowlingTeam", data.teams, "Select bowling team");
    fillSelect("venue", data.venues, "Select stadium / venue");
    $("formMessage").textContent = "";
  } catch (error) {
    $("formMessage").textContent = "Backend not connected yet. The dashboard needs /api/options and /api/predict endpoints to use your trained Python models.";
  }
}

function renderScoreComparison(rfScore, xgbScore) {
  const rf = Math.round(Number(rfScore));
  const xgb = Math.round(Number(xgbScore));
  setText("compareRfScore", `${rf} runs`);
  setText("compareXgbScore", `${xgb} runs`);
  setText("scoreDifference", `${Math.abs(rf - xgb)} runs`);
  const canvas = $("scoreComparisonChart");
  if (!canvas || typeof Chart === "undefined") return;
  if (scoreComparisonChart) scoreComparisonChart.destroy();
  scoreComparisonChart = new Chart(canvas, {
    type: "bar",
    data: {
      labels: ["Random Forest", "XGBoost"],
      datasets: [{
        label: "Predicted final score (runs)",
        data: [rf, xgb],
        backgroundColor: ["#7669eb", "#31b58d"],
        borderRadius: 8,
        maxBarThickness: 64
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: context => `${context.parsed.y} runs` } }
      },
      scales: {
        y: { beginAtZero: true, title: { display: true, text: "Predicted runs" }, grid: { color: "#f0f1f6" } },
        x: { grid: { display: false } }
      }
    }
  });
}

$("predictionForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const battingTeam = $("battingTeam").value;
  const bowlingTeam = $("bowlingTeam").value;
  const venue = $("venue").value;
  const currentOver = Number($("currentOver").value);
  const currentWickets = Number($("currentWickets").value);
  const currentRuns = Number($("currentRuns").value);
  if (battingTeam === bowlingTeam) {
    $("formMessage").textContent = "Choose different batting and bowling teams.";
    return;
  }
  if (currentOver < 0 || currentOver >= 20 || currentWickets < 0 || currentWickets > 9 || currentRuns < 0) {
    $("formMessage").textContent = "Check the over (0–19.5), wickets (0–9), and current runs.";
    return;
  }
  $("formMessage").textContent = "Calculating predictions and retrieving historical statistics…";
  try {
    const result = await getJSON("/api/predict", {
      method: "POST",
      body: JSON.stringify({ batting_team: battingTeam, bowling_team: bowlingTeam, venue, current_over: currentOver, current_wickets: currentWickets, current_runs: currentRuns })
    });
    const rf = Number(result.predictions.random_forest);
    const xgb = Number(result.predictions.xgboost);
    setText("rfScore", Math.round(rf)); setText("xgbScore", Math.round(xgb));
    setText("rfRange", `${Math.max(currentRuns, Math.round(rf - 10))}–${Math.round(rf + 10)} runs`);
    setText("xgbRange", `${Math.max(currentRuns, Math.round(xgb - 10))}–${Math.round(xgb + 10)} runs`);
    const avg = Math.round((rf + xgb) / 2);
    setText("avgScore", avg); setText("projectionText", `${avg} runs projected`);
    $("progressBar").style.width = `${Math.min(100, Math.max(8, avg / 300 * 100))}%`;
    renderScoreComparison(rf, xgb);
    if (result.stats) renderStats(result.stats);
    $("formMessage").textContent = "Predictions updated successfully.";
  } catch (error) {
    $("formMessage").textContent = `Could not calculate prediction: ${error.message}. Ensure the Flask/Python backend is running and its model features match the submitted inputs.`;
  }
});

function renderStats(stats) {
  setText("matchesPlayed", stats.matches_played ?? "—");
  setText("batWins", stats.batting_team_wins ?? "—");
  setText("bowlWins", stats.bowling_team_wins ?? "—");
  setText("averageScore", stats.average_first_innings_score ?? "—");
  const rows = stats.recent_matches || [];
  $("historyMessage").textContent = rows.length ? "Recent matches involving the selected teams." : "No matching historical matches were returned by the backend.";
  const tbody = $("historyTable").querySelector("tbody");
  tbody.innerHTML = "";
  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty-cell">No historical matches found.</td></tr>';
    return;
  }
  rows.forEach(match => {
    const tr = document.createElement("tr");
    [match.date_or_season, match.venue, match.batting_team_score, match.bowling_team_score, match.winner].forEach(value => {
      const td = document.createElement("td"); td.textContent = value ?? "—"; tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}
loadOptions();
