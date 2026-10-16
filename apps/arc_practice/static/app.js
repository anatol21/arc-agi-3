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
  benchmarkModels: { ollama: [], openrouter: [] },
  batchPollTimer: null,
  batchModels: [],
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
  benchmarkProvider: document.querySelector("#benchmark-provider"),
  benchmarkModelLabel: document.querySelector("#benchmark-model-label"),
  benchmarkModel: document.querySelector("#benchmark-model"),
  refreshModels: document.querySelector("#refresh-models"),
  openRouterSettings: document.querySelectorAll(".openrouter-setting"),
  openRouterApiKey: document.querySelector("#openrouter-api-key"),
  openRouterModelId: document.querySelector("#openrouter-model-id"),
  addOpenRouterModel: document.querySelector("#add-openrouter-model"),
  benchmarkParallelRequests: document.querySelector("#benchmark-parallel-requests"),
  benchmarkOrder: document.querySelector("#benchmark-order"),
  benchmarkTaskCount: document.querySelector("#benchmark-task-count"),
  benchmarkSeed: document.querySelector("#benchmark-seed"),
  benchmarkTemperature: document.querySelector("#benchmark-temperature"),
  benchmarkTopP: document.querySelector("#benchmark-top-p"),
  benchmarkNumPredict: document.querySelector("#benchmark-num-predict"),
  benchmarkTimeout: document.querySelector("#benchmark-timeout"),
  benchmarkThink: document.querySelector("#benchmark-think"),
  reasoningNote: document.querySelector("#reasoning-note"),
  benchmarkCoreInstruction: document.querySelector("#benchmark-core-instruction"),
  benchmarkModifier: document.querySelector("#benchmark-prompt-modifier"),
  startBenchmark: document.querySelector("#start-benchmark"),
  cancelBenchmark: document.querySelector("#cancel-benchmark"),
  exportJson: document.querySelector("#export-json"),
  exportCsv: document.querySelector("#export-csv"),
  benchmarkMessage: document.querySelector("#benchmark-message"),
  benchmarkState: document.querySelector("#benchmark-state"),
  benchmarkProgress: document.querySelector("#benchmark-progress"),
  benchmarkRequestProgress: document.querySelector("#benchmark-request-progress"),
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
  benchmarkProviderResponse: document.querySelector("#benchmark-provider-response"),
  batchForm: document.querySelector("#experiment-batch-form"),
  batchDataset: document.querySelector("#batch-dataset"),
  batchSource: document.querySelector("#batch-source"),
  batchOpenRouterApiKey: document.querySelector("#batch-openrouter-api-key"),
  batchRefreshModels: document.querySelector("#batch-refresh-models"),
  batchModelOptions: document.querySelector("#batch-model-options"),
  batchTaskCount: document.querySelector("#batch-task-count"),
  batchSeed: document.querySelector("#batch-seed"),
  batchTrialsPerTask: document.querySelector("#batch-trials-per-task"),
  batchGlobalParallel: document.querySelector("#batch-global-parallel"),
  batchExperiments: document.querySelector("#batch-experiments"),
  addExperiment: document.querySelector("#add-experiment"),
  startExperimentBatch: document.querySelector("#start-experiment-batch"),
  cancelExperimentBatch: document.querySelector("#cancel-experiment-batch"),
  exportBatchJson: document.querySelector("#export-batch-json"),
  exportBatchCsv: document.querySelector("#export-batch-csv"),
  batchMessage: document.querySelector("#batch-message"),
  batchState: document.querySelector("#batch-state"),
  batchProgress: document.querySelector("#batch-progress"),
  batchRequestProgress: document.querySelector("#batch-request-progress"),
  batchActiveRequests: document.querySelector("#batch-active-requests"),
  batchCohortSize: document.querySelector("#batch-cohort-size"),
  batchScorecards: document.querySelector("#batch-scorecards"),
  batchTaskMatrix: document.querySelector("#batch-task-matrix"),
  batchResultsBody: document.querySelector("#batch-results-body"),
  batchInspector: document.querySelector("#batch-inspector"),
  closeBatchInspector: document.querySelector("#close-batch-inspector"),
  batchInspectorSummary: document.querySelector("#batch-inspector-summary"),
  batchPrediction: document.querySelector("#batch-prediction"),
  batchRawResponse: document.querySelector("#batch-raw-response"),
  batchReasoning: document.querySelector("#batch-reasoning"),
  batchAttempts: document.querySelector("#batch-attempts"),
  batchModelRequest: document.querySelector("#batch-model-request"),
  batchProviderResponse: document.querySelector("#batch-provider-response"),
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
  if (!response.ok) {
    // Static assets are read on every request, whereas Python route definitions
    // are loaded only when the local server starts. This makes a stale server
    // otherwise look like a mysterious, bare 404 after the batch UI is updated.
    if (response.status === 404 && path.startsWith("/api/benchmark/batch/")) {
      throw new Error("Parallel experiment batches are not available in the running ARC Practice server. Restart ARC Practice from this folder, then reload this page.");
    }
    throw new Error(body.error || "Request failed.");
  }
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

function setBatchMessage(message = "", kind = "") {
  elements.batchMessage.textContent = message;
  elements.batchMessage.className = `benchmark-message ${kind}`;
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

function renderBatchSources() {
  populateSourceSelect(elements.batchSource, datasetById(elements.batchDataset.value));
}

function batchInput(type, value, attributes = {}) {
  const input = document.createElement("input");
  input.type = type;
  if (type === "checkbox") input.checked = Boolean(value);
  else input.value = value;
  Object.entries(attributes).forEach(([key, attributeValue]) => {
    if (attributeValue !== null) input.setAttribute(key, String(attributeValue));
  });
  return input;
}

function batchField(labelText, control, fieldName) {
  const label = document.createElement("label");
  label.textContent = labelText;
  control.dataset.field = fieldName;
  label.append(control);
  return label;
}

function defaultExperiment(number) {
  return {
    label: `Experiment ${number}`,
    model: "",
    temperature: "0",
    top_p: "1",
    num_predict: "16384",
    timeout_seconds: "120",
    think: false,
    core_instruction: "Solve this ARC task. Infer the transformation from the training examples and apply it to every test input.",
    prompt_modifier: "",
  };
}

function experimentCardValue(card) {
  const value = (field) => card.querySelector(`[data-field="${field}"]`);
  return {
    label: value("label").value,
    model: value("model").value,
    temperature: value("temperature").value,
    top_p: value("top_p").value,
    num_predict: value("num_predict").value,
    timeout_seconds: value("timeout_seconds").value,
    think: value("think").checked,
    core_instruction: value("core_instruction").value,
    prompt_modifier: value("prompt_modifier").value,
  };
}

function refreshExperimentCardActions() {
  const cards = [...elements.batchExperiments.querySelectorAll(".experiment-card")];
  elements.addExperiment.disabled = cards.length >= 5;
  cards.forEach((card, index) => {
    card.querySelector(".experiment-card-title").textContent = `Configuration ${index + 1}`;
    card.querySelector(".remove-experiment").disabled = cards.length === 1;
  });
}

function addExperimentCard(config = null) {
  const values = { ...defaultExperiment(elements.batchExperiments.children.length + 1), ...(config || {}) };
  const card = document.createElement("article");
  card.className = "experiment-card";
  const header = document.createElement("div");
  header.className = "experiment-card-header";
  const heading = document.createElement("h5");
  heading.className = "experiment-card-title";
  const actions = document.createElement("div");
  actions.className = "experiment-card-actions";
  const duplicate = document.createElement("button");
  duplicate.type = "button";
  duplicate.textContent = "Duplicate";
  duplicate.addEventListener("click", () => {
    if (elements.batchExperiments.children.length < 5) addExperimentCard(experimentCardValue(card));
  });
  const remove = document.createElement("button");
  remove.type = "button";
  remove.className = "remove-experiment";
  remove.textContent = "Remove";
  remove.addEventListener("click", () => {
    if (elements.batchExperiments.children.length > 1) {
      card.remove();
      refreshExperimentCardActions();
    }
  });
  actions.append(duplicate, remove);
  header.append(heading, actions);
  const fields = document.createElement("div");
  fields.className = "experiment-fields";
  fields.append(
    batchField("Name", batchInput("text", values.label, { maxlength: 128 }), "label"),
    batchField("OpenRouter model", batchInput("text", values.model, { list: "batch-model-options", maxlength: 512, placeholder: "provider/model" }), "model"),
    batchField("Temperature", batchInput("number", values.temperature, { min: 0, max: 2, step: 0.05 }), "temperature"),
    batchField("Top-p", batchInput("number", values.top_p, { min: 0, max: 1, step: 0.05 }), "top_p"),
    batchField("Maximum tokens", batchInput("number", values.num_predict, { min: 1, max: 131072 }), "num_predict"),
    batchField("Timeout (seconds)", batchInput("number", values.timeout_seconds, { min: 1, max: 600 }), "timeout_seconds"),
  );
  const thinking = batchInput("checkbox", values.think);
  const thinkingLabel = batchField("Request reasoning trace", thinking, "think");
  thinkingLabel.className = "experiment-thinking";
  fields.append(thinkingLabel);
  const promptFields = document.createElement("div");
  promptFields.className = "experiment-prompts";
  const core = document.createElement("textarea");
  core.rows = 3;
  core.maxLength = 32768;
  core.value = values.core_instruction;
  const modifier = document.createElement("textarea");
  modifier.rows = 3;
  modifier.maxLength = 32768;
  modifier.value = values.prompt_modifier;
  promptFields.append(
    batchField("Core instruction", core, "core_instruction"),
    batchField("Prompt modifier", modifier, "prompt_modifier"),
  );
  card.append(header, fields, promptFields);
  elements.batchExperiments.append(card);
  refreshExperimentCardActions();
}

function initializeExperimentCards() {
  if (elements.batchExperiments.children.length) return;
  for (let number = 1; number <= 5; number += 1) addExperimentCard(defaultExperiment(number));
}

function batchPayload() {
  const experiments = [...elements.batchExperiments.querySelectorAll(".experiment-card")].map(experimentCardValue);
  return {
    dataset_id: elements.batchDataset.value,
    source: elements.batchSource.value || null,
    openrouter_api_key: elements.batchOpenRouterApiKey.value,
    task_count: elements.batchTaskCount.value,
    seed: elements.batchSeed.value,
    trials_per_task: elements.batchTrialsPerTask.value,
    global_parallel_requests: elements.batchGlobalParallel.value,
    experiments,
  };
}

async function refreshBatchModels() {
  const apiKey = elements.batchOpenRouterApiKey.value.trim();
  if (!apiKey) {
    setBatchMessage("Enter an OpenRouter API key before loading models.", "error");
    return;
  }
  elements.batchRefreshModels.disabled = true;
  setBatchMessage("Loading OpenRouter models…");
  try {
    const response = await api("/api/benchmark/models", {
      method: "POST",
      body: JSON.stringify({ provider: "openrouter", openrouter_api_key: apiKey }),
    });
    state.batchModels = response.models;
    elements.batchModelOptions.innerHTML = "";
    response.models.forEach((model) => {
      const option = document.createElement("option");
      option.value = model;
      elements.batchModelOptions.append(option);
    });
    setBatchMessage(response.models.length ? `${response.models.length} OpenRouter models available for every configuration.` : "No OpenRouter models were returned.");
  } catch (error) {
    setBatchMessage(error.message, "error");
  } finally {
    elements.batchRefreshModels.disabled = false;
  }
}

function formatSeconds(value) {
  return typeof value === "number" ? `${value.toFixed(3)}s` : "—";
}

function formatRequestProgress(request) {
  const received = request.answer_chars ? `${request.answer_chars.toLocaleString()} answer characters received`
    : request.reasoning_chars ? `${request.reasoning_chars.toLocaleString()} reasoning characters received` : "";
  return `${request.experiment_label ? `${request.experiment_label} · ` : ""}${request.task_id}: ${request.stage} · ${formatSeconds(request.elapsed_seconds)}${received ? ` · ${received}` : ""}`;
}

function appendScoreMetric(container, label, value) {
  const item = document.createElement("div");
  item.className = "scorecard-metric";
  const caption = document.createElement("span");
  caption.textContent = label;
  const amount = document.createElement("strong");
  amount.textContent = value;
  item.append(caption, amount);
  container.append(item);
}

function formatUsage(usage = {}) {
  const totalTokens = usage.total_tokens ?? ((usage.prompt_tokens || 0) + (usage.completion_tokens || 0) + (usage.reasoning_tokens || 0));
  const pieces = [];
  if (totalTokens) pieces.push(`${Number(totalTokens).toLocaleString()} tokens`);
  if (typeof usage.cost === "number") pieces.push(`$${usage.cost.toFixed(4)}`);
  return pieces.length ? pieces.join(" · ") : "Not returned";
}

function renderBatchScorecards(summaries = []) {
  elements.batchScorecards.innerHTML = "";
  if (!summaries.length) {
    elements.batchScorecards.textContent = "No experiment batch has started.";
    return;
  }
  summaries.forEach((summary) => {
    const card = document.createElement("article");
    card.className = "batch-scorecard";
    const heading = document.createElement("h4");
    heading.textContent = summary.label;
    const note = document.createElement("p");
    note.className = "field-note";
    note.textContent = `${summary.completed_trials} of ${summary.target_trials} trials completed`;
    const metrics = document.createElement("div");
    metrics.className = "scorecard-metrics";
    appendScoreMetric(metrics, "Exact trials", `${summary.exact_trials} / ${summary.target_trials}`);
    appendScoreMetric(metrics, "Stable tasks", `${summary.stable_tasks} / ${summary.target_tasks}`);
    appendScoreMetric(metrics, "Passed once", `${summary.pass_once_tasks} / ${summary.target_tasks}`);
    appendScoreMetric(metrics, "Test grids", `${summary.test_grids_correct} / ${summary.target_test_grids}`);
    appendScoreMetric(metrics, "Valid JSON", `${summary.valid_responses} / ${summary.target_trials}`);
    appendScoreMetric(metrics, "Provider replies", `${summary.provider_successes} / ${summary.target_trials}`);
    appendScoreMetric(metrics, "Provider p50 / p95", `${formatSeconds(summary.provider_latency_p50)} / ${formatSeconds(summary.provider_latency_p95)}`);
    appendScoreMetric(metrics, "Queue p50 / p95", `${formatSeconds(summary.queue_seconds_p50)} / ${formatSeconds(summary.queue_seconds_p95)}`);
    appendScoreMetric(metrics, "Usage / cost", formatUsage(summary.usage));
    const failures = Object.entries(summary.status_counts || {}).map(([status, count]) => `${status}: ${count}`).join(" · ") || "No trials completed";
    const failureNote = document.createElement("p");
    failureNote.className = "scorecard-statuses";
    failureNote.textContent = failures;
    card.append(heading, note, metrics, failureNote);
    elements.batchScorecards.append(card);
  });
}

function renderBatchTaskMatrix(snapshot) {
  elements.batchTaskMatrix.innerHTML = "";
  const cohort = snapshot.cohort || [];
  const experiments = snapshot.experiments || [];
  if (!cohort.length || !experiments.length) {
    elements.batchTaskMatrix.textContent = "No batch cohort has started.";
    return;
  }
  const bySlot = new Map();
  const activeBySlot = new Map();
  (snapshot.results || []).forEach((result) => {
    bySlot.set(`${result.task_order}:${result.trial_index}:${result.experiment_id}`, result);
  });
  (snapshot.in_flight || []).forEach((request) => {
    activeBySlot.set(`${request.task_order}:${request.trial_index}:${request.experiment_id}`, request);
  });
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headerRow = document.createElement("tr");
  ["Task", "Trial", ...experiments.map((experiment) => experiment.label)].forEach((label) => {
    const cell = document.createElement("th");
    cell.textContent = label;
    headerRow.append(cell);
  });
  head.append(headerRow);
  const body = document.createElement("tbody");
  cohort.forEach((task) => {
    for (let trialIndex = 0; trialIndex < snapshot.config.trials_per_task; trialIndex += 1) {
      const row = document.createElement("tr");
      const taskCell = document.createElement("td");
      taskCell.textContent = task.task_id;
      const trialCell = document.createElement("td");
      trialCell.textContent = String(trialIndex + 1);
      row.append(taskCell, trialCell);
      experiments.forEach((experiment) => {
        const cell = document.createElement("td");
        const result = bySlot.get(`${task.task_order}:${trialIndex}:${experiment.id}`);
        if (result) {
          const button = document.createElement("button");
          button.type = "button";
          button.className = `matrix-status ${result.status}`;
          button.textContent = result.status;
          button.addEventListener("click", () => inspectBatchResult(result.result_index));
          cell.append(button);
        } else {
          const active = activeBySlot.get(`${task.task_order}:${trialIndex}:${experiment.id}`);
          cell.textContent = active ? `${active.stage} · ${formatSeconds(active.elapsed_seconds)}`
            : snapshot.state === "cancelled" ? "Not run" : "Queued";
        }
        row.append(cell);
      });
      body.append(row);
    }
  });
  table.append(head, body);
  elements.batchTaskMatrix.append(table);
}

function renderBatchResults(results = []) {
  elements.batchResultsBody.innerHTML = "";
  if (!results.length) {
    elements.batchResultsBody.innerHTML = '<tr><td colspan="8">No completed batch trials yet.</td></tr>';
    return;
  }
  results.forEach((result) => {
    const row = document.createElement("tr");
    const values = [
      result.experiment_label,
      result.task_id,
      String((result.trial_index || 0) + 1),
      result.status,
      formatSeconds(result.queue_seconds),
      formatSeconds(result.provider_latency_seconds),
      result.error || "",
    ];
    values.forEach((value) => {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    });
    const inspectCell = document.createElement("td");
    const inspectButton = document.createElement("button");
    inspectButton.className = "inspect-button";
    inspectButton.type = "button";
    inspectButton.textContent = "Inspect";
    inspectButton.addEventListener("click", () => inspectBatchResult(result.result_index));
    inspectCell.append(inspectButton);
    row.append(inspectCell);
    elements.batchResultsBody.append(row);
  });
}

function renderBatchStatus(snapshot) {
  const running = snapshot.state === "running";
  elements.batchState.textContent = snapshot.state === "idle" ? "Idle" : snapshot.state;
  elements.batchProgress.textContent = `${snapshot.completed_trial_count || 0} / ${snapshot.configured_trial_count || 0}`;
  elements.batchActiveRequests.textContent = `${snapshot.active_requests || 0} / ${snapshot.global_parallel_requests || 0}`;
  elements.batchRequestProgress.textContent = (snapshot.in_flight || []).map(formatRequestProgress).join(" · ");
  elements.batchCohortSize.textContent = `${snapshot.cohort?.length || 0} tasks`;
  elements.startExperimentBatch.disabled = running;
  elements.cancelExperimentBatch.disabled = !running;
  elements.addExperiment.disabled = running || elements.batchExperiments.children.length >= 5;
  const canExport = !running && (snapshot.completed_trial_count || 0) > 0;
  elements.exportBatchJson.disabled = !canExport;
  elements.exportBatchCsv.disabled = !canExport;
  renderBatchScorecards(snapshot.experiments || []);
  renderBatchTaskMatrix(snapshot);
  renderBatchResults(snapshot.results || []);
}

async function refreshModels() {
  const provider = elements.benchmarkProvider.value;
  const isOpenRouter = provider === "openrouter";
  const apiKey = elements.openRouterApiKey.value.trim();
  if (isOpenRouter && !apiKey) {
    setBenchmarkMessage("Enter an OpenRouter API key before loading models.", "error");
    return;
  }
  elements.refreshModels.disabled = true;
  setBenchmarkMessage(isOpenRouter ? "Loading OpenRouter models…" : "Finding local Ollama models…");
  try {
    const response = isOpenRouter
      ? await api("/api/benchmark/models", {
        method: "POST",
        body: JSON.stringify({ provider, openrouter_api_key: apiKey }),
      })
      : await api("/api/benchmark/models");
    state.benchmarkModels[provider] = response.models;
    populateSelect(
      elements.benchmarkModel,
      response.models.map((model) => ({ value: model, text: model })),
      response.models[0] || "",
    );
    state.benchmarkModelsLoaded = true;
    const source = isOpenRouter ? "OpenRouter" : "local Ollama";
    setBenchmarkMessage(response.models.length ? `${response.models.length} ${source} model(s) available.` : `No ${source} models are available.`);
  } catch (error) {
    elements.benchmarkModel.innerHTML = `<option value="">${isOpenRouter ? "OpenRouter unavailable" : "Ollama unavailable"}</option>`;
    setBenchmarkMessage(error.message, "error");
  } finally {
    elements.refreshModels.disabled = false;
  }
}

function renderBenchmarkProvider() {
  const isOpenRouter = elements.benchmarkProvider.value === "openrouter";
  elements.openRouterSettings.forEach((setting) => { setting.hidden = !isOpenRouter; });
  elements.benchmarkModelLabel.firstChild.textContent = isOpenRouter ? "OpenRouter model" : "Ollama model";
  elements.refreshModels.textContent = isOpenRouter ? "Load OpenRouter models" : "Refresh local models";
  elements.benchmarkNumPredict.max = isOpenRouter ? "131072" : "32768";
  elements.benchmarkThink.disabled = false;
  if (isOpenRouter) {
    elements.reasoningNote.textContent = "Reasoning and the answer share Maximum tokens. With at least 2,048 maximum tokens, half is requested as the reasoning budget where supported. Returned reasoning is displayed separately.";
  } else {
    elements.reasoningNote.textContent = "Thinking consumes the maximum-token budget. Some models, including Qwen3-Coder, do not provide a separate trace.";
  }
  const models = state.benchmarkModels[isOpenRouter ? "openrouter" : "ollama"];
  populateSelect(
    elements.benchmarkModel,
    models.map((model) => ({ value: model, text: model })),
    models[0] || "",
  );
  state.benchmarkModelsLoaded = models.length > 0;
}

function addOpenRouterModel() {
  const model = elements.openRouterModelId.value.trim();
  if (!model) {
    setBenchmarkMessage("Enter an OpenRouter model ID to add it.", "error");
    return;
  }
  const models = state.benchmarkModels.openrouter;
  if (!models.includes(model)) models.push(model);
  models.sort((first, second) => first.localeCompare(second));
  populateSelect(elements.benchmarkModel, models.map((value) => ({ value, text: value })), model);
  state.benchmarkModelsLoaded = true;
  setBenchmarkMessage(`${model} is selected for OpenRouter.`);
}

function benchmarkPayload() {
  const provider = elements.benchmarkProvider.value;
  return {
    dataset_id: elements.benchmarkDataset.value,
    source: elements.benchmarkSource.value || null,
    provider,
    model: elements.benchmarkModel.value,
    order: elements.benchmarkOrder.value,
    task_count: elements.benchmarkTaskCount.value,
    seed: elements.benchmarkSeed.value,
    temperature: elements.benchmarkTemperature.value,
    top_p: elements.benchmarkTopP.value,
    num_predict: elements.benchmarkNumPredict.value,
    timeout_seconds: elements.benchmarkTimeout.value,
    parallel_requests: provider === "openrouter" ? elements.benchmarkParallelRequests.value : 1,
    think: elements.benchmarkThink.checked,
    core_instruction: elements.benchmarkCoreInstruction.value,
    prompt_modifier: elements.benchmarkModifier.value,
    ...(provider === "openrouter" ? { openrouter_api_key: elements.openRouterApiKey.value } : {}),
  };
}

function renderBenchmarkStatus(snapshot) {
  elements.benchmarkRequestProgress.textContent = (snapshot.in_flight || []).map(formatRequestProgress).join(" · ");
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
  elements.benchmarkProviderResponse.textContent = "";
}

function formattedRawResponse(rawResponse) {
  if (rawResponse == null) return "No model response was received.";
  if (rawResponse === "") return "The provider returned no final answer text. Review the reasoning and provider completion details.";
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
  if (modelRequest == null) return "No model request was constructed for this task.";
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
    elements.benchmarkProviderResponse.textContent = JSON.stringify({ ...result.provider_response, usage: result.usage }, null, 2);
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
  setBenchmarkMessage("Starting benchmark…");
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
    setBenchmarkMessage("Cancellation requested. Active model requests may finish first; queued tasks will not start.");
    pollBenchmark();
  } catch (error) {
    setBenchmarkMessage(error.message, "error");
  }
}

function clearBatchInspector() {
  elements.batchInspector.hidden = true;
  elements.batchInspectorSummary.textContent = "";
  elements.batchPrediction.innerHTML = "";
  elements.batchRawResponse.textContent = "";
  elements.batchReasoning.textContent = "";
  elements.batchAttempts.textContent = "";
  elements.batchModelRequest.textContent = "";
  elements.batchProviderResponse.textContent = "";
}

function renderBatchPrediction(prediction) {
  elements.batchPrediction.innerHTML = "";
  const outputs = prediction?.outputs;
  if (!Array.isArray(outputs) || !outputs.length) {
    elements.batchPrediction.textContent = "No valid parsed prediction. Review the raw response and attempt history.";
    return;
  }
  outputs.forEach((grid, index) => {
    const card = document.createElement("div");
    card.className = "prediction-card";
    const label = document.createElement("p");
    label.textContent = `Predicted output ${index + 1}`;
    card.append(label, createGridElement(grid, { placement: "inspection" }));
    elements.batchPrediction.append(card);
  });
}

async function inspectBatchResult(resultIndex) {
  setBatchMessage("Loading batch trial response…");
  try {
    const result = await api(`/api/benchmark/batch/result?result_index=${encodeURIComponent(resultIndex)}`);
    elements.batchInspector.hidden = false;
    elements.batchInspectorSummary.textContent = `${result.experiment_label} · ${result.task_id} · trial ${(result.trial_index || 0) + 1} · ${result.status} · queue ${formatSeconds(result.queue_seconds)} · provider ${formatSeconds(result.provider_latency_seconds)}${result.error ? ` · ${result.error}` : ""}`;
    renderBatchPrediction(result.prediction);
    elements.batchModelRequest.textContent = formattedModelRequest(result.model_request);
    elements.batchRawResponse.textContent = formattedRawResponse(result.raw_response);
    elements.batchReasoning.textContent = formattedReasoning(result.reasoning);
    elements.batchAttempts.textContent = JSON.stringify(result.attempts || [], null, 2);
    elements.batchProviderResponse.textContent = JSON.stringify({ ...result.provider_response, usage: result.usage }, null, 2);
    elements.batchInspector.scrollIntoView({ behavior: "smooth", block: "nearest" });
    setBatchMessage();
  } catch (error) {
    setBatchMessage(error.message, "error");
  }
}

async function pollExperimentBatch() {
  window.clearTimeout(state.batchPollTimer);
  try {
    const snapshot = await api("/api/benchmark/batch/status");
    renderBatchStatus(snapshot);
    if (snapshot.state === "running") state.batchPollTimer = window.setTimeout(pollExperimentBatch, 1000);
    else if (snapshot.state !== "idle") setBatchMessage(`Batch ${snapshot.state}: ${snapshot.completed_trial_count} of ${snapshot.configured_trial_count} trials processed.`);
  } catch (error) {
    setBatchMessage(error.message, "error");
  }
}

async function startExperimentBatch(event) {
  event.preventDefault();
  clearBatchInspector();
  setBatchMessage("Starting shared-cohort experiment batch…");
  try {
    const snapshot = await api("/api/benchmark/batch/start", { method: "POST", body: JSON.stringify(batchPayload()) });
    renderBatchStatus(snapshot);
    pollExperimentBatch();
  } catch (error) {
    setBatchMessage(error.message, "error");
  }
}

async function cancelExperimentBatch() {
  try {
    const snapshot = await api("/api/benchmark/batch/cancel", { method: "POST", body: JSON.stringify({ cancel: true }) });
    renderBatchStatus(snapshot);
    setBatchMessage("Cancellation requested. In-flight provider calls will finish before the batch stops.");
    pollExperimentBatch();
  } catch (error) {
    setBatchMessage(error.message, "error");
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
  elements.benchmarkProvider.addEventListener("change", () => {
    renderBenchmarkProvider();
    if (elements.benchmarkProvider.value === "ollama") refreshModels();
  });
  elements.refreshModels.addEventListener("click", refreshModels);
  elements.addOpenRouterModel.addEventListener("click", addOpenRouterModel);
  elements.benchmarkForm.addEventListener("submit", startBenchmark);
  elements.cancelBenchmark.addEventListener("click", cancelBenchmark);
  elements.exportJson.addEventListener("click", () => { window.location.assign("/api/benchmark/export?format=json"); });
  elements.exportCsv.addEventListener("click", () => { window.location.assign("/api/benchmark/export?format=csv"); });
  elements.closeBenchmarkInspector.addEventListener("click", clearBenchmarkInspector);
  elements.batchDataset.addEventListener("change", renderBatchSources);
  elements.batchRefreshModels.addEventListener("click", refreshBatchModels);
  elements.addExperiment.addEventListener("click", () => {
    if (elements.batchExperiments.children.length < 5) addExperimentCard();
  });
  elements.batchForm.addEventListener("submit", startExperimentBatch);
  elements.cancelExperimentBatch.addEventListener("click", cancelExperimentBatch);
  elements.exportBatchJson.addEventListener("click", () => { window.location.assign("/api/benchmark/batch/export?format=json"); });
  elements.exportBatchCsv.addEventListener("click", () => { window.location.assign("/api/benchmark/batch/export?format=csv"); });
  elements.closeBatchInspector.addEventListener("click", clearBatchInspector);
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
    populateSelect(elements.batchDataset, choices, registry.default_dataset_id);
    populateSourceSelect(elements.sourceFilter, datasetById(state.manualDatasetId));
    renderBenchmarkSources();
    renderBatchSources();
    initializeExperimentCards();
    renderBenchmarkProvider();
    manualDatasetSummary();
    await loadTask();
    await pollBenchmark();
    await pollExperimentBatch();
  } catch (error) {
    setTaskStatus(`Could not start ARC Practice: ${error.message}`, true);
  }
}

initialize();
