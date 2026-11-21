/* ============================================================
   ChurnLens front-end
   Plain JavaScript. No dependencies, no gradients.
   ============================================================ */

(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  const SVG_NS = "http://www.w3.org/2000/svg";
  const GAUGE_LENGTH = 289.03;
  const DONUT_LENGTH = 314.159;

  const money = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
  const moneyPrecise = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 2,
  });
  const integer = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });
  const oneDecimal = new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });

  const state = {
    meta: null,
    stats: null,
    defaults: {},
    values: {},
    fields: [],
    activeGroup: null,
    loading: false,
  };

  const dom = {
    header: $(".site-header"),
    themeToggle: $("#theme-toggle"),
    form: $("#predict-form"),
    formTabs: $("#form-tabs"),
    formFields: $("#form-fields"),
    fieldCount: $("#field-count"),
    predictBtn: $("#predict-btn"),
    resultCard: $("#result-card"),
    resultEmpty: $("#result-empty"),
    resultContent: $("#result-content"),
    resultTitle: $("#result-title"),
    riskChip: $("#risk-chip"),
    gaugeFill: $("#gauge-fill"),
    resultProb: $("#result-prob"),
    resultBand: $("#result-band"),
    resultPercentile: $("#result-percentile"),
    resultVar: $("#result-var"),
    factorList: $("#factor-list"),
    recList: $("#rec-list"),
    resultNote: $("#result-note"),
    sampleHigh: $("#sample-high"),
    sampleLow: $("#sample-low"),
    resetBtn: $("#reset-btn"),
    footerVersion: $("#footer-version"),
  };

  /* ---------- Formatting ---------- */

  const pct = (value, digits = 1) =>
    `${(Number(value) * 100).toFixed(digits).replace(/\.0$/, "")}%`;

  const signed = (value) => {
    const n = Number(value);
    if (!Number.isFinite(n) || Math.abs(n) < 0.05) return "0.0 pts";
    return `${n > 0 ? "+" : "−"}${Math.abs(n).toFixed(1)} pts`;
  };

  const ordinal = (n) => {
    const value = Math.max(0, Math.min(100, Math.round(Number(n) || 0)));
    const suffix =
      value % 10 === 1 && value !== 11
        ? "st"
        : value % 10 === 2 && value !== 12
          ? "nd"
          : value % 10 === 3 && value !== 13
            ? "rd"
            : "th";
    return `${value}${suffix}`;
  };

  const setText = (id, text) => {
    const element = document.getElementById(id);
    if (element) element.textContent = text;
  };

  /* ---------- Theme ---------- */

  const applyTheme = (theme) => {
    document.documentElement.dataset.theme = theme;
    const isDark = theme === "dark";
    dom.themeToggle.setAttribute(
      "aria-label",
      isDark ? "Switch to light mode" : "Switch to dark mode"
    );
    dom.themeToggle.setAttribute("title", isDark ? "Switch to light mode" : "Switch to dark mode");
    const metaTheme = $('meta[name="theme-color"]');
    if (metaTheme) metaTheme.setAttribute("content", isDark ? "#0b100e" : "#f4f5f1");
  };

  const initTheme = () => {
    let theme;
    try {
      theme = localStorage.getItem("churnlens-theme");
    } catch {
      theme = null;
    }
    if (theme !== "light" && theme !== "dark") {
      theme = window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    }
    applyTheme(theme);

    dom.themeToggle.addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      applyTheme(next);
      try {
        localStorage.setItem("churnlens-theme", next);
      } catch {
        /* storage may be unavailable */
      }
    });
  };

  /* ---------- Reveal + header ---------- */

  const initMotion = () => {
    const items = $$(".reveal");
    if (!("IntersectionObserver" in window)) {
      items.forEach((item) => item.classList.add("is-visible"));
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1, rootMargin: "0px 0px -36px 0px" }
    );
    items.forEach((item) => observer.observe(item));

    const onScroll = () => dom.header.classList.toggle("is-scrolled", window.scrollY > 8);
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  };

  /* ---------- Toast ---------- */

  let toastTimer = null;
  const toast = (message) => {
    let element = $(".toast");
    if (!element) {
      element = document.createElement("div");
      element.className = "toast";
      element.setAttribute("role", "status");
      document.body.appendChild(element);
    }
    element.textContent = message;
    element.classList.add("is-visible");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => element.classList.remove("is-visible"), 4200);
  };

  /* ---------- Data loading ---------- */

  const fetchJSON = async (url, options) => {
    const response = await fetch(url, options);
    if (!response.ok) {
      let detail = `Request failed (${response.status})`;
      try {
        const payload = await response.json();
        if (Array.isArray(payload.detail)) {
          detail = payload.detail
            .map((item) => item.msg || item.message || String(item))
            .join(", ");
        } else if (payload.detail) {
          detail = payload.detail;
        }
      } catch {
        /* ignore non-JSON errors */
      }
      throw new Error(detail);
    }
    return response.json();
  };

  const init = async () => {
    initTheme();
    initMotion();

    dom.sampleHigh.disabled = true;
    dom.sampleLow.disabled = true;
    dom.resetBtn.disabled = true;
    dom.fieldCount.textContent = "Loading signals…";

    try {
      const [meta, stats] = await Promise.all([
        fetchJSON("/api/meta"),
        fetchJSON("/api/stats"),
      ]);
      state.meta = meta;
      state.stats = stats.stats;
      state.fields = meta.fields;

      state.defaults = {};
      meta.fields.forEach((field) => {
        state.defaults[field.name] = field.default;
      });
      state.values = { ...state.defaults };

      renderOverview(meta, stats.stats);
      buildForm(meta);
      renderCharts(stats.stats, meta.model.metrics);

      dom.sampleHigh.disabled = false;
      dom.sampleLow.disabled = false;
      dom.resetBtn.disabled = false;

      const trained = new Date(meta.model.version);
      dom.footerVersion.textContent = Number.isNaN(trained.getTime())
        ? "Gradient Boosting model"
        : `Gradient Boosting · trained ${trained.toLocaleDateString(undefined, {
            day: "numeric",
            month: "short",
            year: "numeric",
          })}`;
    } catch (error) {
      console.error(error);
      dom.fieldCount.textContent = "Could not load the model";
      dom.formFields.innerHTML =
        '<div class="load-error">The API could not be reached. Make sure the FastAPI server is running, then refresh.</div>';
      toast("Could not load model metadata. Is the FastAPI server running?");
    }
  };

  /* ---------- Overview ---------- */

  const renderOverview = (meta, stats) => {
    const overview = stats.overview;
    const metrics = meta.model.metrics;

    setText("hero-f1", pct(metrics.f1));
    setText("hero-auc", Number(metrics.roc_auc).toFixed(3));
    setText("hero-churn-rate", pct(overview.churn_rate));
    setText("hero-churned", integer.format(overview.churned));
    setText("hero-retained", integer.format(overview.retained));
    setText("hero-total", integer.format(overview.total_customers));
    setText("hero-ltv", money.format(overview.avg_lifetime_value));
    setText("hero-tenure", `${overview.avg_tenure.toFixed(1)} yrs`);

    setText("kpi-churn", pct(overview.churn_rate));
    setText("kpi-retained", integer.format(overview.retained));
    setText("kpi-ltv", money.format(overview.avg_lifetime_value));
    setText("kpi-accuracy", pct(metrics.accuracy));
    setText("kpi-recall", pct(metrics.recall));

    const donut = $("#hero-donut");
    if (donut) {
      requestAnimationFrame(() => {
        donut.style.strokeDashoffset = String(
          DONUT_LENGTH * (1 - Math.min(1, Math.max(0, overview.churn_rate)))
        );
      });
    }
  };

  /* ---------- Form ---------- */

  const renderCurrentGroup = () => {
    const group = state.activeGroup;
    const fields = state.fields.filter((field) => field.group === group);
    const fragment = document.createDocumentFragment();
    const grid = document.createElement("div");
    grid.className = "field-grid";

    fields.forEach((field) => {
      const wrapper = document.createElement("div");
      wrapper.className = "field";

      const label = document.createElement("label");
      label.setAttribute("for", `field-${field.name}`);
      const labelText = document.createElement("span");
      labelText.textContent = field.label;
      label.appendChild(labelText);
      if (field.unit) {
        const unit = document.createElement("span");
        unit.className = "unit";
        unit.textContent = field.unit;
        label.appendChild(unit);
      }

      let input;
      if (field.type === "select") {
        input = document.createElement("select");
        field.options.forEach((option) => {
          const optionElement = document.createElement("option");
          optionElement.value = option;
          optionElement.textContent = option;
          input.appendChild(optionElement);
        });
        input.value = String(state.values[field.name] ?? field.default);
      } else {
        input = document.createElement("input");
        input.type = "number";
        input.inputMode = "decimal";
        if (field.min !== null && field.min !== undefined) input.min = field.min;
        if (field.max !== null && field.max !== undefined) input.dataset.max = field.max;
        // Numeric values in this dataset are continuous, so never enforce a
        // step increment. A step like 0.1 anchors to min and rejects valid
        // values such as 13.6 when min is 0.24.
        input.step = "any";
        input.value = String(state.values[field.name] ?? field.default ?? "");
      }

      input.id = `field-${field.name}`;
      input.name = field.name;
      input.dataset.field = field.name;
      input.addEventListener("input", () => {
        state.values[field.name] = input.value;
        wrapper.classList.remove("is-invalid");
      });
      input.addEventListener("change", () => {
        state.values[field.name] = input.value;
      });

      wrapper.append(label, input);
      grid.appendChild(wrapper);
    });

    fragment.appendChild(grid);
    dom.formFields.replaceChildren(fragment);
    dom.formTabs.querySelectorAll(".tab").forEach((tab) => {
      tab.setAttribute("aria-selected", String(tab.dataset.group === group));
    });
    dom.formFields.setAttribute("aria-labelledby", `tab-${slug(group)}`);
  };

  const slug = (value) => value.toLowerCase().replace(/[^a-z0-9]+/g, "-");

  const buildForm = (meta) => {
    const groups = meta.groups.filter((group) =>
      meta.fields.some((field) => field.group === group)
    );
    state.activeGroup = groups[0];
    dom.formTabs.replaceChildren();
    dom.formFields.id = "form-fields";
    dom.formFields.setAttribute("role", "tabpanel");
    dom.formFields.setAttribute("tabindex", "0");

    groups.forEach((group) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "tab";
      button.dataset.group = group;
      button.id = `tab-${slug(group)}`;
      button.setAttribute("role", "tab");
      button.setAttribute("aria-controls", "form-fields");
      button.textContent = group;
      button.addEventListener("click", () => {
        state.activeGroup = group;
        renderCurrentGroup();
      });
      dom.formTabs.appendChild(button);
    });

    dom.formTabs.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      const tabs = $$(".tab", dom.formTabs);
      const current = tabs.indexOf(document.activeElement);
      if (current === -1) return;
      event.preventDefault();
      const nextIndex =
        event.key === "ArrowRight"
          ? (current + 1) % tabs.length
          : (current - 1 + tabs.length) % tabs.length;
      tabs[nextIndex].focus();
      tabs[nextIndex].click();
    });

    dom.fieldCount.textContent = `${meta.field_count} signals · ${groups.length} groups`;
    renderCurrentGroup();
  };

  const collectPayload = () => {
    const payload = {};
    state.fields.forEach((field) => {
      const raw = state.values[field.name];
      if (field.type === "number") {
        const number = Number(raw);
        if (raw === "" || raw === null || raw === undefined || Number.isNaN(number)) {
          payload[field.name] = field.default;
        } else {
          payload[field.name] = number;
        }
      } else {
        payload[field.name] = raw || field.default;
      }
    });
    return payload;
  };

  const setLoading = (loading) => {
    state.loading = loading;
    dom.predictBtn.disabled = loading;
    dom.predictBtn.classList.toggle("is-loading", loading);
    dom.predictBtn.querySelector(".btn-label").textContent = loading
      ? "Scoring customer…"
      : "Assess churn risk";
  };

  const resetResult = () => {
    dom.resultContent.hidden = true;
    dom.resultEmpty.hidden = false;
    dom.resultCard.classList.remove("risk-low", "risk-medium", "risk-high");
  };

  const submitPrediction = async (event) => {
    if (event) event.preventDefault();
    if (state.loading || !state.meta) return;

    if (!dom.form.checkValidity()) {
      dom.form.reportValidity();
      return;
    }

    setLoading(true);
    try {
      const prediction = await fetchJSON("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectPayload()),
      });
      renderResult(prediction);
    } catch (error) {
      console.error(error);
      toast(error.message || "Could not score this customer.");
    } finally {
      setLoading(false);
    }
  };

  /* ---------- Result ---------- */

  const renderResult = (data) => {
    const bandClass =
      data.risk_band === "High" ? "high" : data.risk_band === "Elevated" ? "medium" : "low";

    dom.resultEmpty.hidden = true;
    dom.resultContent.hidden = false;
    dom.resultCard.classList.remove("risk-low", "risk-medium", "risk-high");
    dom.resultCard.classList.add(`risk-${bandClass}`);

    dom.resultTitle.textContent = data.label;
    dom.riskChip.textContent = data.risk_band;
    dom.riskChip.className = `risk-chip ${bandClass}`;
    dom.resultProb.textContent = pct(data.probability, 1);
    dom.resultBand.textContent = data.risk_band;
    dom.resultPercentile.textContent = ordinal(data.percentile);
    dom.resultVar.textContent = money.format(data.value_at_risk);

    requestAnimationFrame(() => {
      dom.gaugeFill.style.strokeDashoffset = String(
        GAUGE_LENGTH * (1 - Math.min(1, Math.max(0, data.probability)))
      );
    });

    dom.factorList.replaceChildren();
    const maxImpact = Math.max(
      1,
      ...data.factors.map((factor) => Math.abs(Number(factor.impact) || 0))
    );
    data.factors.forEach((factor) => {
      const direction =
        factor.direction === "up" ? "up" : factor.direction === "down" ? "down" : "neutral";
      const row = document.createElement("div");
      row.className = "factor";

      const top = document.createElement("div");
      top.className = "factor-top";
      const name = document.createElement("span");
      name.className = "factor-name";
      name.textContent = factor.label;
      const impact = document.createElement("span");
      impact.className = `factor-impact ${direction}`;
      impact.textContent = signed(factor.impact);
      top.append(name, impact);

      const meta = document.createElement("div");
      meta.className = "factor-meta";
      const current = document.createElement("span");
      current.textContent = `This customer: ${factor.value}`;
      const typical = document.createElement("span");
      typical.textContent = `Typical: ${factor.typical}`;
      meta.append(current, typical);

      const bar = document.createElement("div");
      bar.className = "factor-bar";
      const fill = document.createElement("span");
      fill.className = direction;
      requestAnimationFrame(() => {
        fill.style.width = `${Math.max(4, (Math.abs(Number(factor.impact) || 0) / maxImpact) * 100)}%`;
      });
      bar.appendChild(fill);

      row.append(top, meta, bar);
      dom.factorList.appendChild(row);
    });

    dom.recList.replaceChildren();
    data.recommendations.forEach((recommendation) => {
      const item = document.createElement("li");
      item.textContent = recommendation;
      dom.recList.appendChild(item);
    });

    const threshold = pct(data.threshold, 0);
    dom.resultNote.textContent = `Scored at a ${threshold} decision threshold. This score is higher than ${data.percentile}% of held-out customers. Lifetime value at risk = probability × ${moneyPrecise.format(
      data.lifetime_value
    )}.`;
  };

  /* ---------- Sample / reset ---------- */

  const loadSample = (kind) => {
    const sample = state.meta?.samples?.[kind];
    if (!sample) {
      toast("Sample profile is not available.");
      return;
    }
    state.values = { ...state.defaults, ...sample };
    state.activeGroup = state.meta.groups.find((group) =>
      state.meta.fields.some((field) => field.group === group)
    );
    state.fields.forEach((field) => {
      if (sample[field.name] !== undefined && sample[field.name] !== null) {
        state.values[field.name] = sample[field.name];
      }
    });
    renderCurrentGroup();
    submitPrediction();
  };

  const resetForm = () => {
    state.values = { ...state.defaults };
    state.fields.forEach((field) => {
      state.values[field.name] = field.default;
    });
    renderCurrentGroup();
    resetResult();
    toast("Form reset to the typical customer.");
  };

  /* ---------- Charts ---------- */

  const barRow = ({ label, value, max, display, color, title }) => {
    const row = document.createElement("div");
    row.className = "bar-row";

    const labelElement = document.createElement("span");
    labelElement.className = "bar-label";
    labelElement.textContent = label;
    labelElement.title = title ? `${label} — ${title}` : label;

    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    if (color) fill.style.background = color;
    track.appendChild(fill);

    const valueElement = document.createElement("span");
    valueElement.className = "bar-value";
    valueElement.textContent = display;

    row.append(labelElement, track, valueElement);
    requestAnimationFrame(() => {
      fill.style.width = `${Math.max(2, (value / max) * 100)}%`;
    });
    return row;
  };

  const renderCountryChart = (rows) => {
    const container = $("#country-chart");
    container.replaceChildren();
    const max = Math.max(...rows.map((row) => row.rate), 0.01);
    rows.forEach((row) => {
      container.appendChild(
        barRow({
          label: row.label,
          value: row.rate,
          max,
          display: pct(row.rate),
          color: "var(--chart-4)",
          title: `${integer.format(row.customers)} customers`,
        })
      );
    });
  };

  const svgElement = (tag, attributes = {}) => {
    const element = document.createElementNS(SVG_NS, tag);
    Object.entries(attributes).forEach(([key, value]) => {
      element.setAttribute(key, String(value));
    });
    return element;
  };

  const renderTenureChart = (rows) => {
    const container = $("#tenure-chart");
    container.replaceChildren();

    const width = 440;
    const height = 196;
    const margin = { top: 22, right: 16, bottom: 32, left: 34 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const maxRate = Math.max(...rows.map((row) => row.rate), 0.01) * 1.18;

    const svg = svgElement("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": "Churn rate as customer service calls increase",
    });
    const title = svgElement("title");
    title.textContent = "Line chart showing churn rate rises sharply as customer service calls increase.";
    svg.appendChild(title);

    [0.5, 1].forEach((fraction) => {
      const y = margin.top + plotHeight * (1 - fraction);
      svg.appendChild(
        svgElement("line", {
          x1: margin.left,
          y1: y,
          x2: width - margin.right,
          y2: y,
          class: "line-grid",
        })
      );
      const label = svgElement("text", {
        x: margin.left - 8,
        y: y + 3,
        "text-anchor": "end",
        class: "line-label",
      });
      label.textContent = `${Math.round(maxRate * fraction * 100)}%`;
      svg.appendChild(label);
    });

    const baseline = svgElement("text", {
      x: margin.left - 8,
      y: margin.top + plotHeight + 3,
      "text-anchor": "end",
      class: "line-label",
    });
    baseline.textContent = "0%";
    svg.appendChild(baseline);

    const points = rows.map((row, index) => {
      const x =
        margin.left + (rows.length === 1 ? plotWidth / 2 : (index / (rows.length - 1)) * plotWidth);
      const y = margin.top + plotHeight * (1 - row.rate / maxRate);
      return { x, y, row };
    });

    const linePath = points
      .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
      .join(" ");
    const areaPath = `${linePath} L ${points[points.length - 1].x} ${
      margin.top + plotHeight
    } L ${points[0].x} ${margin.top + plotHeight} Z`;

    svg.appendChild(svgElement("path", { d: areaPath, class: "line-area" }));
    svg.appendChild(svgElement("path", { d: linePath, class: "line-path" }));

    points.forEach((point) => {
      svg.appendChild(
        svgElement("circle", { cx: point.x, cy: point.y, r: 4.1, class: "line-dot" })
      );
      const value = svgElement("text", {
        x: point.x,
        y: point.y - 10,
        class: "line-value",
      });
      value.textContent = pct(point.row.rate);
      svg.appendChild(value);

      const label = svgElement("text", {
        x: point.x,
        y: height - 8,
        class: "line-label",
      });
      label.textContent = point.row.label;
      svg.appendChild(label);
    });

    container.appendChild(svg);
  };

  const renderRiskChart = (rows) => {
    const container = $("#risk-chart");
    container.replaceChildren();
    const max = Math.max(...rows.map((row) => row.count), 1);
    const total = rows.reduce((sum, row) => sum + row.count, 0);

    rows.forEach((row, index) => {
      const center = (index + 0.5) * 10;
      const band = center < 35 ? "low" : center < 65 ? "medium" : "high";
      const bar = document.createElement("div");
      bar.className = `histogram-bar ${band}`;
      bar.title = `${row.count} customers`;

      const fill = document.createElement("i");
      fill.style.height = `${Math.max(2, (row.count / max) * 100)}%`;
      const tip = document.createElement("em");
      tip.className = "bar-tip";
      tip.textContent = `${row.count} (${total ? ((row.count / total) * 100).toFixed(1) : "0"}%)`;
      const label = document.createElement("span");
      label.textContent = `${row.label}%`;

      bar.append(tip, fill, label);
      container.appendChild(bar);

      requestAnimationFrame(() => {
        fill.style.transform = "scaleY(1)";
      });
    });
  };

  const renderImportanceChart = (rows) => {
    const container = $("#importance-chart");
    container.replaceChildren();
    const max = Math.max(...rows.map((row) => row.importance), 0.0001);
    rows.forEach((row) => {
      container.appendChild(
        barRow({
          label: row.label,
          value: row.importance,
          max,
          display: `${(row.importance * 100).toFixed(1)}%`,
          color: "var(--chart-1)",
          title: `${(row.importance * 100).toFixed(2)}% of model importance`,
        })
      );
    });
  };

  const renderRocChart = (points, auc) => {
    const container = $("#roc-chart");
    container.replaceChildren();

    const width = 440;
    const height = 206;
    const margin = { top: 18, right: 18, bottom: 30, left: 36 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;

    const svg = svgElement("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": `Receiving operating characteristic curve with AUC ${auc}`,
    });
    const title = svgElement("title");
    title.textContent = `ROC curve. AUC equals ${auc}.`;
    svg.appendChild(title);

    [0, 0.5, 1].forEach((fraction) => {
      const y = margin.top + plotHeight * (1 - fraction);
      const x = margin.left + plotWidth * fraction;
      svg.appendChild(
        svgElement("line", {
          x1: margin.left,
          y1: y,
          x2: width - margin.right,
          y2: y,
          class: "line-grid",
        })
      );
      svg.appendChild(
        svgElement("line", {
          x1: x,
          y1: margin.top,
          x2: x,
          y2: margin.top + plotHeight,
          class: "line-grid",
        })
      );
      const yLabel = svgElement("text", {
        x: margin.left - 8,
        y: y + 3,
        "text-anchor": "end",
        class: "line-label",
      });
      yLabel.textContent = `${Math.round(fraction * 100)}%`;
      svg.appendChild(yLabel);
      const xLabel = svgElement("text", {
        x,
        y: height - 9,
        class: "line-label",
      });
      xLabel.textContent = `${Math.round(fraction * 100)}%`;
      svg.appendChild(xLabel);
    });

    svg.appendChild(
      svgElement("line", {
        x1: margin.left,
        y1: margin.top + plotHeight,
        x2: width - margin.right,
        y2: margin.top,
        class: "roc-diagonal",
      })
    );

    const projected = points.map((point) => ({
      x: margin.left + point.fpr * plotWidth,
      y: margin.top + (1 - point.tpr) * plotHeight,
    }));
    const line = projected
      .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x} ${point.y}`)
      .join(" ");
    const area = `${line} L ${margin.left + plotWidth} ${margin.top + plotHeight} L ${
      margin.left
    } ${margin.top + plotHeight} Z`;

    svg.appendChild(svgElement("path", { d: area, class: "roc-area" }));
    svg.appendChild(svgElement("path", { d: line, class: "roc-line" }));

    container.appendChild(svg);
    setText("auc-value", Number(auc).toFixed(3));
  };

  const renderBenchmark = (results) => {
    const container = $("#benchmark-chart");
    container.replaceChildren();

    const legend = document.createElement("div");
    legend.className = "chart-legend benchmark-legend";
    legend.innerHTML =
      '<span><i class="swatch" style="background: var(--chart-2); opacity: .6"></i> Accuracy</span>' +
      '<span><i class="swatch" style="background: var(--chart-1)"></i> F1 score</span>';
    container.appendChild(legend);

    const rows = Object.entries(results).sort((a, b) => b[1].f1 - a[1].f1);
    rows.forEach(([name, metrics]) => {
      const row = document.createElement("div");
      row.className = "benchmark-row";

      const label = document.createElement("span");
      label.className = "benchmark-name";
      label.textContent = name;

      const track = document.createElement("div");
      track.className = "benchmark-track";

      const accuracyBar = document.createElement("div");
      accuracyBar.className = "benchmark-bar accuracy";
      const accuracyFill = document.createElement("span");
      accuracyBar.appendChild(accuracyFill);

      const f1Bar = document.createElement("div");
      f1Bar.className = "benchmark-bar";
      const f1Fill = document.createElement("span");
      f1Bar.appendChild(f1Fill);
      track.append(accuracyBar, f1Bar);

      const value = document.createElement("span");
      value.className = "benchmark-value";
      value.textContent = pct(metrics.f1);
      value.title = `Accuracy ${pct(metrics.accuracy)} · F1 ${pct(metrics.f1)}`;

      row.append(label, track, value);
      container.appendChild(row);

      requestAnimationFrame(() => {
        accuracyFill.style.width = `${metrics.accuracy * 100}%`;
        f1Fill.style.width = `${metrics.f1 * 100}%`;
      });
    });
  };

  const renderCharts = (stats, metrics) => {
    renderCountryChart(stats.churn_by_country);
    renderTenureChart(stats.churn_by_support || stats.churn_by_tenure);
    renderRiskChart(stats.risk_histogram);
    renderImportanceChart(stats.grouped_importance);
    renderRocChart(stats.roc, stats.roc_auc);
    renderBenchmark(stats.model_comparison);
  };

  /* ---------- Wiring ---------- */

  dom.form.addEventListener("submit", submitPrediction);
  dom.sampleHigh.addEventListener("click", () => loadSample("high_risk"));
  dom.sampleLow.addEventListener("click", () => loadSample("low_risk"));
  dom.resetBtn.addEventListener("click", resetForm);

  init();
})();
