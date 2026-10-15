"use strict";

const state = {
  datasets: [],
  datasetsById: new Map(),
  manualDatasetId: "",
  task: null,
  testIndex: 0,
  outputGrid: makeGrid(3, 3),
  selectedColor: 0,
  tool: "edit",
  selected: new Set(),
  selectionStart: null,
  dragging: false,
  painting: false,
  clipboard: [],
  attempts: 0,
  solved: 0,
  solvedTests: new Set(),
  benchmarkPollTimer: null,
  benchmarkModelsLoaded: false,
};

const elements = {
  datasetSummary: document.querySelector("#dataset-summary"),
  manualView: document.querySelector("#manual-view"),
  benchmarkView: document.querySelector("#benchmark-view"),
  manualViewButton: document.querySelector("#manual-view-button"),
  benchmarkViewButton: document.querySelector("#benchmark-view-button"),
  manualDataset: document.querySelector("#manual-dataset-filter"),
  sourceFilter: document.querySelector("#source-filter"),
  taskStatus: document.querySelector("#task-status"),
  taskIdForm: document.querySelector("#task-id-form"),
  taskIdInput: document.querySelector("#task-id-input"),
  randomTask: document.querySelector("#random-task"),
  previousTask: document.querySelector("#previous-task"),
  nextTask: document.querySelector("#next-task"),
  workspace: document.querySelector("#workspace"),
  demos: document.querySelector("#demonstrations"),
  demoCount: document.querySelector("#demo-count"),
  testInput: document.querySelector("#test-input"),
  testCount: document.querySelector("#test-count"),
  previousTest: document.querySelector("#previous-test"),
  nextTest: document.querySelector("#next-test"),
  outputGrid: document.querySelector("#output-grid"),
  gridSizeInput: document.querySelector("#grid-size-input"),
  gridSizeDisplay: document.querySelector("#grid-size-display"),
  resizeGrid: document.querySelector("#resize-grid"),
  copyInput: document.querySelector("#copy-input"),
  resetGrid: document.querySelector("#reset-grid"),
  submitGrid: document.querySelector("#submit-grid"),
  palette: document.querySelector("#palette"),
  showNumbers: document.querySelector("#show-numbers"),
  solutionStatus: document.querySelector("#solution-status"),
  scoreValue: document.querySelector("#score-value"),
  benchmarkForm: document.querySelector("#benchmark-form"),
  benchmarkDataset: document.querySelector("#benchmark-dataset"),
  benchmarkSource: document.querySelector("#benchmark-source"),
  benchmarkModel: document.querySelector("#benchmark-model"),
  refreshModels: document.querySelector("#refresh-models"),
  benchmarkOrder: document.querySelector("#benchmark-order"),
  benchmarkTaskCount: document.querySelector("#benchmark-task-count"),
  benchmarkSeed: document.querySelector("#benchmark-seed"),
  benchmarkTemperature: document.querySelector("#benchmark-temperature"),
  benchmarkTopP: document.querySelector("#benchmark-top-p"),
  benchmarkNumPredict: document.querySelector("#benchmark-num-predict"),
  benchmarkTimeout: document.querySelector("#benchmark-timeout"),
  benchmarkCoreInstruction: document.querySelector("#benchmark-core-instruction"),
  benchmarkModifier: document.querySelector("#benchmark-prompt-modifier"),
  startBenchmark: document.querySelector("#start-benchmark"),
  cancelBenchmark: document.querySelector("#cancel-benchmark"),
  exportJson: document.querySelector("#export-json"),
  exportCsv: document.querySelector("#export-csv"),
  benchmarkMessage: document.querySelector("#benchmark-message"),
  benchmarkState: document.querySelector("#benchmark-state"),
  benchmarkProgress: document.querySelector("#benchmark-progress"),
  benchmarkTaskScore: document.querySelector("#benchmark-task-score"),
  benchmarkTestScore: document.querySelector("#benchmark-test-score"),
  benchmarkResultsBody: document.querySelector("#benchmark-results-body"),
  benchmarkInspector: document.querySelector("#benchmark-inspector"),
  closeBenchmarkInspector: document.querySelector("#close-benchmark-inspector"),
  benchmarkInspectorSummary: document.querySelector("#benchmark-inspector-summary"),
  benchmarkPrediction: document.querySelector("#benchmark-prediction"),
  benchmarkRawResponse: document.querySelector("#benchmark-raw-response"),
  benchmarkReasoning: document.querySelector("#benchmark-reasoning"),
  benchmarkModelRequest: document.querySelector("#benchmark-model-request"),
};

function makeGrid(rows, columns, previous = null) {
  return Array.from({ length: rows }, (_, row) =>
    Array.from({ length: columns }, (_, column) => previous?.[row]?.[column] ?? 0)
  );
}

function cloneGrid(grid) {
  return grid.map((row) => [...row]);
}

function cellKey(row, column) {
  return `${row}:${column}`;
}

function keyToCoordinates(key) {
  return key.split(":").map(Number);
}

function currentTest() {
  return state.task.test[state.testIndex];
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const body = await response.json().catch(() => ({ error: "The server returned invalid JSON." }));
  if (!response.ok) throw new Error(body.error || "Request failed.");
  return body;
}

function setTaskStatus(message, isError = false) {
  elements.taskStatus.textContent = message;
  elements.taskStatus.classList.toggle("error", isError);
}

function setSolutionStatus(message = "", kind = "") {
  elements.solutionStatus.textContent = message;
  elements.solutionStatus.className = `solution-status ${kind}`;
}

function setBenchmarkMessage(message = "", kind = "") {
  elements.benchmarkMessage.textContent = message;
  elements.benchmarkMessage.className = `benchmark-message ${kind}`;
}

function updateScore() {
  elements.scoreValue.textContent = `${state.solved} solved / ${state.attempts} attempts`;
}

function datasetById(datasetId) {
  return state.datasetsById.get(datasetId);
}

function populateSelect(select, choices, selectedValue) {
  select.innerHTML = "";
  choices.forEach(({ value, text }) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = text;
    select.append(option);
  });
  select.value = selectedValue;
}

function populateSourceSelect(select, dataset, selectedValue = "") {
  const choices = [{ value: "", text: "All sources" }];
  Object.entries(dataset.sources).forEach(([source, count]) => {
    choices.push({ value: source, text: `${source} (${count.toLocaleString()})` });
  });
  populateSelect(select, choices, selectedValue);
}

function manualDatasetSummary() {
  const dataset = datasetById(state.manualDatasetId);
  if (!dataset) return;
  const available = dataset.oversize_task_count
    ? `${dataset.playable_task_count.toLocaleString()} eligible of ${dataset.task_count.toLocaleString()} · ${dataset.oversize_task_count.toLocaleString()} over 30×30 excluded`
    : `${dataset.task_count.toLocaleString()} eligible tasks`;
  const evaluation = dataset.evaluation_only ? " · evaluation-only" : "";
  elements.datasetSummary.textContent = `${dataset.label} · ${available}${evaluation} · answers stay local`;
}

function renderPalette() {
  elements.palette.innerHTML = "";
  for (let color = 0; color <= 9; color += 1) {
    const button = document.createElement("button");
    button.className = `palette-button color-${color}${color === state.selectedColor ? " active" : ""}`;
    button.type = "button";
    button.title = `Color ${color}`;
    button.setAttribute("aria-label", `Select color ${color}`);
    button.addEventListener("click", () => {
      state.selectedColor = color;
      if (state.tool === "select" && state.selected.size) applyColorToSelection();
      renderPalette();
      renderEditor();
    });
    elements.palette.append(button);
  }
}

function createGridElement(grid, { editable = false, placement = "test" } = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = "arc-grid";
  wrapper.style.setProperty("--cols", grid[0].length);
  const largestSide = Math.max(grid.length, grid[0].length);
  const sizing = placement === "demo"
    ? { minimum: 10, maximum: 32, target: 320 }
    : placement === "editor"
      ? { minimum: 12, maximum: 44, target: 560 }
      : placement === "inspection"
        ? { minimum: 12, maximum: 28, target: 360 }
      : { minimum: 12, maximum: 42, target: 460 };
  const cellSize = Math.max(sizing.minimum, Math.min(sizing.maximum, Math.floor(sizing.target / largestSide)));
  wrapper.style.setProperty("--cell-size", `${cellSize}px`);
  wrapper.dataset.placement = placement;
  grid.forEach((row, rowIndex) => row.forEach((color, columnIndex) => {
    const cell = document.createElement("div");
    cell.className = `grid-cell color-${color}`;
    cell.dataset.row = rowIndex;
    cell.dataset.column = columnIndex;
    cell.setAttribute("role", editable ? "button" : "img");
    cell.setAttribute("aria-label", `row ${rowIndex + 1}, column ${columnIndex + 1}, color ${color}`);
    if (elements.showNumbers.checked) cell.textContent = String(color);
    if (editable && state.selected.has(cellKey(rowIndex, columnIndex))) cell.classList.add("selected");
    wrapper.append(cell);
  }));
  return wrapper;
}

function renderDemonstrations() {
  elements.demos.innerHTML = "";
  elements.demoCount.textContent = `${state.task.train.length} examples`;
  state.task.train.forEach((pair, index) => {
    const card = document.createElement("article");
    card.className = "demo-card";
    card.innerHTML = `<span class="demo-label">Example ${index + 1}</span>`;
    const pairContainer = document.createElement("div");
    pairContainer.className = "demo-pair";
    const input = document.createElement("div");
    input.className = "demo-grid";
    input.innerHTML = '<div class="grid-label">Input</div>';
    const inputFrame = document.createElement("div");
    inputFrame.className = "demo-grid-frame";
    inputFrame.append(createGridElement(pair.input, { placement: "demo" }));
    input.append(inputFrame);
    const arrow = document.createElement("div");
    arrow.className = "pair-arrow";
    arrow.textContent = "→";
    const output = document.createElement("div");
    output.className = "demo-grid";
    output.innerHTML = '<div class="grid-label">Output</div>';
    const outputFrame = document.createElement("div");
    outputFrame.className = "demo-grid-frame";
    outputFrame.append(createGridElement(pair.output, { placement: "demo" }));
    output.append(outputFrame);
    pairContainer.append(input, arrow, output);
    card.append(pairContainer);
    elements.demos.append(card);
  });
}

function renderTestInput() {
  elements.testInput.innerHTML = "";
  elements.testInput.append(createGridElement(currentTest().input, { placement: "test" }));
  elements.testCount.textContent = `${state.testIndex + 1} / ${state.task.test_count}`;
  elements.previousTest.disabled = state.testIndex === 0;
  elements.nextTest.disabled = state.testIndex >= state.task.test_count - 1;
}

function renderEditor() {
  elements.outputGrid.innerHTML = "";
  elements.outputGrid.dataset.mode = state.tool;
  elements.outputGrid.append(createGridElement(state.outputGrid, { editable: true, placement: "editor" }));
  elements.gridSizeDisplay.textContent = `${state.outputGrid.length} × ${state.outputGrid[0].length}`;
  elements.gridSizeInput.value = `${state.outputGrid.length}x${state.outputGrid[0].length}`;
}

function renderTask() {
  elements.workspace.hidden = false;
  elements.taskIdInput.value = state.task.task_id;
  setTaskStatus(`${datasetById(state.task.dataset_id).label} · task ${state.task.task_id} · ${state.task.source} · record ${state.task.position + 1}`);
  renderDemonstrations();
  renderTestInput();
  renderEditor();
  renderPalette();
}

function resetOutputGrid() {
  state.outputGrid = makeGrid(3, 3);
  state.selected.clear();
  renderEditor();
}

function selectTest(index) {
  if (index < 0 || index >= state.task.test_count) return;
  state.testIndex = index;
  resetOutputGrid();
  renderTestInput();
  setSolutionStatus();
}

async function loadTask({ mode = "random", taskId = "" } = {}) {
  try {
    setTaskStatus("Loading task…");
    const params = new URLSearchParams({ dataset: state.manualDatasetId });
    if (taskId) params.set("task_id", taskId);
    else {
      params.set("mode", mode);
      if (state.task?.dataset_id === state.manualDatasetId) params.set("position", state.task.position);
      if (elements.sourceFilter.value) params.set("source", elements.sourceFilter.value);
    }
    state.task = await api(`/api/task?${params}`);
    state.testIndex = 0;
    state.selected.clear();
    resetOutputGrid();
    setSolutionStatus();
    renderTask();
  } catch (error) {
    setTaskStatus(error.message, true);
  }
}

function parseSize() {
  const match = elements.gridSizeInput.value.trim().match(/^(\d+)\s*x\s*(\d+)$/i);
  if (!match) throw new Error("Use a grid size like 3x3 or 5x7.");
  const rows = Number(match[1]);
  const columns = Number(match[2]);
  if (rows < 1 || columns < 1 || rows > 30 || columns > 30) throw new Error("Grid sides must be between 1 and 30.");
  return [rows, columns];
}

function resizeGrid() {
  try {
    const [rows, columns] = parseSize();
    state.outputGrid = makeGrid(rows, columns, state.outputGrid);
    state.selected.clear();
    renderEditor();
    setSolutionStatus();
  } catch (error) {
    setSolutionStatus(error.message, "error");
  }
}

function paintCell(row, column, color = state.selectedColor) {
  if (state.outputGrid[row][column] !== color) {
    state.outputGrid[row][column] = color;
    renderEditor();
  }
}

function floodFill(row, column) {
  const target = state.outputGrid[row][column];
  if (target === state.selectedColor) return;
  const stack = [[row, column]];
  while (stack.length) {
    const [currentRow, currentColumn] = stack.pop();
    if (currentRow < 0 || currentColumn < 0 || currentRow >= state.outputGrid.length || currentColumn >= state.outputGrid[0].length || state.outputGrid[currentRow][currentColumn] !== target) continue;
    state.outputGrid[currentRow][currentColumn] = state.selectedColor;
    stack.push([currentRow - 1, currentColumn], [currentRow + 1, currentColumn], [currentRow, currentColumn - 1], [currentRow, currentColumn + 1]);
  }
  renderEditor();
}

function selectRectangle(start, end) {
  state.selected.clear();
  for (let row = Math.min(start.row, end.row); row <= Math.max(start.row, end.row); row += 1) {
    for (let column = Math.min(start.column, end.column); column <= Math.max(start.column, end.column); column += 1) state.selected.add(cellKey(row, column));
  }
  renderEditor();
}

function applyColorToSelection() {
  state.selected.forEach((key) => {
    const [row, column] = keyToCoordinates(key);
    state.outputGrid[row][column] = state.selectedColor;
  });
}

function cellFromEvent(event) {
  const cell = event.target.closest(".grid-cell");
  return cell && elements.outputGrid.contains(cell) ? { row: Number(cell.dataset.row), column: Number(cell.dataset.column) } : null;
}

function setupEditorEvents() {
  elements.outputGrid.addEventListener("pointerdown", (event) => {
    const cell = cellFromEvent(event);
    if (!cell) return;
    event.preventDefault();
    if (state.tool === "edit") {
      state.painting = true;
      paintCell(cell.row, cell.column);
    } else if (state.tool === "floodfill") floodFill(cell.row, cell.column);
    else {
      state.dragging = true;
      state.selectionStart = cell;
      selectRectangle(cell, cell);
    }
    elements.outputGrid.setPointerCapture?.(event.pointerId);
  });
  elements.outputGrid.addEventListener("pointermove", (event) => {
    const cell = cellFromEvent(event);
    if (!cell) return;
    if (state.painting && state.tool === "edit") paintCell(cell.row, cell.column);
    if (state.dragging && state.tool === "select" && state.selectionStart) selectRectangle(state.selectionStart, cell);
  });
  const finishPointer = () => { state.dragging = false; state.painting = false; state.selectionStart = null; };
  elements.outputGrid.addEventListener("pointerup", finishPointer);
  elements.outputGrid.addEventListener("pointercancel", finishPointer);
}

function copySelection() {
  if (!state.selected.size) return setSolutionStatus("Select one or more output cells before copying.", "error");
  const cells = [...state.selected].map((key) => {
    const [row, column] = keyToCoordinates(key);
    return { row, column, value: state.outputGrid[row][column] };
  });
  const minRow = Math.min(...cells.map((cell) => cell.row));
  const minColumn = Math.min(...cells.map((cell) => cell.column));
  state.clipboard = cells.map((cell) => ({ ...cell, row: cell.row - minRow, column: cell.column - minColumn }));
  setSolutionStatus(`${state.clipboard.length} cell${state.clipboard.length === 1 ? "" : "s"} copied.`);
}

function pasteSelection() {
  if (!state.clipboard.length) return setSolutionStatus("Copy cells before pasting.", "error");
  if (state.selected.size !== 1) return setSolutionStatus("Select exactly one output cell as the paste destination.", "error");
  const [baseRow, baseColumn] = [...state.selected].map(keyToCoordinates)[0];
  state.clipboard.forEach((cell) => {
    const row = baseRow + cell.row;
    const column = baseColumn + cell.column;
    if (state.outputGrid[row]?.[column] !== undefined) state.outputGrid[row][column] = cell.value;
  });
  renderEditor();
  setSolutionStatus("Pasted copied cells.");
}

async function submitGrid() {
  if (!state.task) return;
  try {
    const result = await api("/api/attempt", {
      method: "POST",
      body: JSON.stringify({ dataset_id: state.task.dataset_id, task_id: state.task.task_id, test_index: state.testIndex, grid: state.outputGrid }),
    });
    state.attempts += 1;
    const solvedKey = `${state.task.dataset_id}:${state.task.task_id}:${state.testIndex}`;
    if (result.correct && !state.solvedTests.has(solvedKey)) { state.solvedTests.add(solvedKey); state.solved += 1; }
    updateScore();
    setSolutionStatus(result.message, result.correct ? "correct" : "error");
  } catch (error) {
    setSolutionStatus(error.message, "error");
  }
}

function renderBenchmarkSources() {
  populateSourceSelect(elements.benchmarkSource, datasetById(elements.benchmarkDataset.value));
}

async function refreshModels() {
  elements.refreshModels.disabled = true;
  setBenchmarkMessage("Finding local Ollama models…");
  try {
    const response = await api("/api/benchmark/models");
    populateSelect(elements.benchmarkModel, response.models.map((model) => ({ value: model, text: model })), response.models[0] || "");
    state.benchmarkModelsLoaded = true;
    setBenchmarkMessage(response.models.length ? `${response.models.length} local model(s) available.` : "No Ollama models are installed.");
  } catch (error) {
    elements.benchmarkModel.innerHTML = '<option value="">Ollama unavailable</option>';
    setBenchmarkMessage(error.message, "error");
  } finally {
    elements.refreshModels.disabled = false;
  }
}

function benchmarkPayload() {
  return {
    dataset_id: elements.benchmarkDataset.value,
    source: elements.benchmarkSource.value || null,
    model: elements.benchmarkModel.value,
    order: elements.benchmarkOrder.value,
    task_count: elements.benchmarkTaskCount.value,
    seed: elements.benchmarkSeed.value,
    temperature: elements.benchmarkTemperature.value,
    top_p: elements.benchmarkTopP.value,
    num_predict: elements.benchmarkNumPredict.value,
    timeout_seconds: elements.benchmarkTimeout.value,
    core_instruction: elements.benchmarkCoreInstruction.value,
    prompt_modifier: elements.benchmarkModifier.value,
  };
}

function renderBenchmarkStatus(snapshot) {
  elements.benchmarkState.textContent = snapshot.state === "idle" ? "Idle" : snapshot.state;
  elements.benchmarkProgress.textContent = `${snapshot.completed_task_count || 0} / ${snapshot.configured_task_count || 0}`;
  elements.benchmarkTaskScore.textContent = String(snapshot.task_correct_count || 0);
  elements.benchmarkTestScore.textContent = `${snapshot.test_correct_count || 0} / ${snapshot.test_total_count || 0}`;
  const running = snapshot.state === "running";
  elements.startBenchmark.disabled = running;
  elements.cancelBenchmark.disabled = !running;
  const canExport = !running && (snapshot.completed_task_count || 0) > 0;
  elements.exportJson.disabled = !canExport;
  elements.exportCsv.disabled = !canExport;
  elements.benchmarkResultsBody.innerHTML = "";
  const results = snapshot.results || [];
  if (!results.length) {
    elements.benchmarkResultsBody.innerHTML = '<tr><td colspan="7">No completed tasks yet.</td></tr>';
    return;
  }
  results.forEach((result) => {
    const row = document.createElement("tr");
    const values = [
      result.task_id,
      result.source,
      result.status,
      result.test_count == null ? "—" : `${result.tests_correct} / ${result.test_count}`,
      result.latency_seconds == null ? "—" : result.latency_seconds.toFixed(3),
      result.error || "",
    ];
    values.forEach((value) => { const cell = document.createElement("td"); cell.textContent = String(value); row.append(cell); });
    const inspectCell = document.createElement("td");
    const inspectButton = document.createElement("button");
    inspectButton.className = "inspect-button";
    inspectButton.type = "button";
    inspectButton.textContent = "Inspect";
    inspectButton.addEventListener("click", () => inspectBenchmarkResult(result.result_index));
    inspectCell.append(inspectButton);
    row.append(inspectCell);
    elements.benchmarkResultsBody.append(row);
  });
}

function clearBenchmarkInspector() {
  elements.benchmarkInspector.hidden = true;
  elements.benchmarkInspectorSummary.textContent = "";
  elements.benchmarkPrediction.innerHTML = "";
  elements.benchmarkRawResponse.textContent = "";
  elements.benchmarkReasoning.textContent = "";
  elements.benchmarkModelRequest.textContent = "";
}

function formattedRawResponse(rawResponse) {
  if (rawResponse == null) return "No model response was received.";
  try {
    return JSON.stringify(JSON.parse(rawResponse), null, 2);
  } catch {
    return rawResponse;
  }
}

function formattedReasoning(reasoning) {
  return reasoning == null || reasoning === ""
    ? "This model did not return a separate reasoning trace. Any reasoning embedded in its response appears above."
    : reasoning;
}

function formattedModelRequest(modelRequest) {
  if (modelRequest == null) return "No Ollama request was constructed for this task.";
  return JSON.stringify(modelRequest, null, 2);
}

function renderPrediction(prediction) {
  elements.benchmarkPrediction.innerHTML = "";
  const outputs = prediction?.outputs;
  if (!Array.isArray(outputs) || !outputs.length) {
    elements.benchmarkPrediction.textContent = "No valid parsed prediction. Review the raw response and parser message.";
    return;
  }
  outputs.forEach((grid, index) => {
    const card = document.createElement("div");
    card.className = "prediction-card";
    const label = document.createElement("p");
    label.textContent = `Predicted output ${index + 1}`;
    card.append(label, createGridElement(grid, { placement: "inspection" }));
    elements.benchmarkPrediction.append(card);
  });
}

async function inspectBenchmarkResult(resultIndex) {
  setBenchmarkMessage("Loading model response…");
  try {
    const result = await api(`/api/benchmark/result?result_index=${encodeURIComponent(resultIndex)}`);
    elements.benchmarkInspector.hidden = false;
    elements.benchmarkInspectorSummary.textContent = `Task ${result.task_id} · ${result.source} · ${result.status} · ${result.latency_seconds.toFixed(3)} seconds${result.error ? ` · ${result.error}` : ""}`;
    renderPrediction(result.prediction);
    elements.benchmarkModelRequest.textContent = formattedModelRequest(result.model_request);
    elements.benchmarkRawResponse.textContent = formattedRawResponse(result.raw_response);
    elements.benchmarkReasoning.textContent = formattedReasoning(result.reasoning);
    elements.benchmarkInspector.scrollIntoView({ behavior: "smooth", block: "nearest" });
    setBenchmarkMessage();
  } catch (error) {
    setBenchmarkMessage(error.message, "error");
  }
}

async function pollBenchmark() {
  window.clearTimeout(state.benchmarkPollTimer);
  try {
    const snapshot = await api("/api/benchmark/status");
    renderBenchmarkStatus(snapshot);
    if (snapshot.state === "running") state.benchmarkPollTimer = window.setTimeout(pollBenchmark, 1000);
    else if (snapshot.state !== "idle") setBenchmarkMessage(`Run ${snapshot.state}: ${snapshot.completed_task_count} task(s) processed.`);
  } catch (error) {
    setBenchmarkMessage(error.message, "error");
  }
}

async function startBenchmark(event) {
  event.preventDefault();
  clearBenchmarkInspector();
  setBenchmarkMessage("Starting local benchmark…");
  try {
    const snapshot = await api("/api/benchmark/start", { method: "POST", body: JSON.stringify(benchmarkPayload()) });
    renderBenchmarkStatus(snapshot);
    pollBenchmark();
  } catch (error) {
    setBenchmarkMessage(error.message, "error");
  }
}

async function cancelBenchmark() {
  try {
    const snapshot = await api("/api/benchmark/cancel", { method: "POST", body: JSON.stringify({ cancel: true }) });
    renderBenchmarkStatus(snapshot);
    setBenchmarkMessage("Cancellation requested. The current Ollama call may finish first.");
    pollBenchmark();
  } catch (error) {
    setBenchmarkMessage(error.message, "error");
  }
}

function showView(view) {
  const benchmark = view === "benchmark";
  elements.manualView.hidden = benchmark;
  elements.benchmarkView.hidden = !benchmark;
  elements.manualViewButton.classList.toggle("active", !benchmark);
  elements.benchmarkViewButton.classList.toggle("active", benchmark);
  elements.manualViewButton.setAttribute("aria-selected", String(!benchmark));
  elements.benchmarkViewButton.setAttribute("aria-selected", String(benchmark));
  if (benchmark && !state.benchmarkModelsLoaded) refreshModels();
}

function registerEvents() {
  elements.manualViewButton.addEventListener("click", () => showView("manual"));
  elements.benchmarkViewButton.addEventListener("click", () => showView("benchmark"));
  elements.manualDataset.addEventListener("change", () => {
    state.manualDatasetId = elements.manualDataset.value;
    state.task = null;
    populateSourceSelect(elements.sourceFilter, datasetById(state.manualDatasetId));
    manualDatasetSummary();
    loadTask();
  });
  elements.sourceFilter.addEventListener("change", () => loadTask());
  elements.randomTask.addEventListener("click", () => loadTask({ mode: "random" }));
  elements.previousTask.addEventListener("click", () => loadTask({ mode: "previous" }));
  elements.nextTask.addEventListener("click", () => loadTask({ mode: "next" }));
  elements.taskIdForm.addEventListener("submit", (event) => { event.preventDefault(); if (elements.taskIdInput.value.trim()) loadTask({ taskId: elements.taskIdInput.value.trim() }); });
  elements.previousTest.addEventListener("click", () => selectTest(state.testIndex - 1));
  elements.nextTest.addEventListener("click", () => selectTest(state.testIndex + 1));
  elements.resizeGrid.addEventListener("click", resizeGrid);
  elements.gridSizeInput.addEventListener("keydown", (event) => { if (event.key === "Enter") resizeGrid(); });
  elements.copyInput.addEventListener("click", () => { state.outputGrid = cloneGrid(currentTest().input); state.selected.clear(); renderEditor(); setSolutionStatus("Copied the test input into your output grid."); });
  elements.resetGrid.addEventListener("click", () => { resetOutputGrid(); setSolutionStatus("Output grid reset to 3x3."); });
  elements.submitGrid.addEventListener("click", submitGrid);
  elements.showNumbers.addEventListener("change", () => { renderDemonstrations(); renderTestInput(); renderEditor(); });
  document.querySelectorAll('input[name="tool"]').forEach((input) => input.addEventListener("change", () => { state.tool = input.value; state.selected.clear(); renderEditor(); setSolutionStatus(state.tool === "select" ? "Drag across output cells to select them." : ""); }));
  document.addEventListener("keydown", (event) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement?.tagName)) return;
    if (event.key.toLowerCase() === "c") { event.preventDefault(); copySelection(); }
    if (event.key.toLowerCase() === "v") { event.preventDefault(); pasteSelection(); }
  });
  elements.benchmarkDataset.addEventListener("change", renderBenchmarkSources);
  elements.refreshModels.addEventListener("click", refreshModels);
  elements.benchmarkForm.addEventListener("submit", startBenchmark);
  elements.cancelBenchmark.addEventListener("click", cancelBenchmark);
  elements.exportJson.addEventListener("click", () => { window.location.assign("/api/benchmark/export?format=json"); });
  elements.exportCsv.addEventListener("click", () => { window.location.assign("/api/benchmark/export?format=csv"); });
  elements.closeBenchmarkInspector.addEventListener("click", clearBenchmarkInspector);
  setupEditorEvents();
}

async function initialize() {
  registerEvents();
  renderPalette();
  try {
    const registry = await api("/api/datasets");
    state.datasets = registry.datasets;
    state.datasetsById = new Map(registry.datasets.map((dataset) => [dataset.id, dataset]));
    const choices = registry.datasets.map((dataset) => ({ value: dataset.id, text: dataset.label }));
    state.manualDatasetId = registry.default_dataset_id;
    populateSelect(elements.manualDataset, choices, state.manualDatasetId);
    populateSelect(elements.benchmarkDataset, choices, registry.default_dataset_id);
    populateSourceSelect(elements.sourceFilter, datasetById(state.manualDatasetId));
    renderBenchmarkSources();
    manualDatasetSummary();
    await loadTask();
    await pollBenchmark();
  } catch (error) {
    setTaskStatus(`Could not start ARC Practice: ${error.message}`, true);
  }
}

initialize();
