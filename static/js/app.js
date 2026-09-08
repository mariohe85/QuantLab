(() => {
  function renderChart(element, rows, title, type = "line") {
    if (!element || !window.echarts || !rows.length) return;
    const chart = window.echarts.getInstanceByDom(element) || window.echarts.init(element);
    const valueFormat = element.dataset.valueFormat;
    const percentage = valueFormat === "percent";
    const percentPoints = valueFormat === "percent-points";
    const percentLike = percentage || percentPoints;
    const scale = percentage ? 100 : 1;
    const suffix = "%";
    // Blank readings must break the line rather than plot as zero.
    const values = rows.map((row) => {
      const value = Number(row.dataset.value);
      if (row.dataset.value === "" || !Number.isFinite(value)) return null;
      return value * scale;
    });
    chart.setOption({
      backgroundColor: "transparent",
      title: { text: element.dataset.title || title, textStyle: { color: "#e8edf7", fontSize: 14 } },
      tooltip: {
        trigger: "axis",
        valueFormatter: percentLike ? (value) => `${Number(value).toFixed(2)}${suffix}` : undefined,
      },
      grid: { left: 52, right: 20, top: 54, bottom: 72 },
      xAxis: { type: "category", data: rows.map((row) => row.dataset.name), axisLabel: { color: "#9aa8c3", rotate: 35 }, axisLine: { lineStyle: { color: "#2b3655" } } },
      yAxis: {
        type: "value",
        axisLabel: { color: "#9aa8c3", formatter: percentLike ? `{value}${suffix}` : "{value}" },
        splitLine: { lineStyle: { color: "#202b47" } },
      },
      series: [{
        type,
        data: type === "bar" ? values.map((value) => ({ value, itemStyle: { color: value < 0 ? "#d8798c" : "#67e8c9" } })) : values,
        showSymbol: false,
      }],
    });
    requestAnimationFrame(() => chart.resize());
    }

  function renderHistoryChart(kind) {
    const element = document.querySelector(`[data-chart="history-${kind}"]`);
    const rows = [...document.querySelectorAll("[data-history-series]")];
    if (!element || !window.echarts || !rows.length) return;
    const equity = kind === "equity";
    const portfolioKey = equity ? "portfolioEquity" : "portfolioDrawdown";
    const spyKey = equity ? "spyEquity" : "spyDrawdown";
    const chart = window.echarts.getInstanceByDom(element) || window.echarts.init(element);
    chart.setOption({
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        valueFormatter: equity ? undefined : (value) => formatPercent(value, 1),
      },
      legend: { top: 8, textStyle: { color: "#9aa8c3" } },
      grid: { left: 58, right: 22, top: 48, bottom: 58 },
      xAxis: {
        type: "category",
        data: rows.map((row) => row.dataset.date),
        axisLabel: { color: "#9aa8c3", hideOverlap: true },
        axisLine: { lineStyle: { color: "#2b3655" } },
      },
      yAxis: {
        type: "value",
        axisLabel: {
          color: "#9aa8c3",
          formatter: equity ? "{value}" : (value) => `${(value * 100).toFixed(0)}%`,
        },
        splitLine: { lineStyle: { color: "#202b47" } },
      },
      series: [
        {
          name: "Portfolio",
          type: "line",
          data: rows.map((row) => Number(row.dataset[portfolioKey])),
          showSymbol: false,
          lineStyle: { color: "#67e8c9", width: 2 },
        },
        {
          name: "SPY",
          type: "line",
          data: rows.map((row) => Number(row.dataset[spyKey])),
          showSymbol: false,
          lineStyle: { color: "#8da2d6", width: 2 },
        },
      ],
    });
    requestAnimationFrame(() => chart.resize());
  }

  function renderOptimizerHistoryChart(kind) {
    const element = document.querySelector(`[data-chart="optimizer-history-${kind}"]`);
    const rows = [...document.querySelectorAll("[data-optimizer-history-series]")];
    if (!element || !window.echarts || !rows.length) return;
    const equity = kind === "equity";
    const keys = equity
      ? ["originalEquity", "optimizedEquity", "spyEquity"]
      : ["originalDrawdown", "optimizedDrawdown", "spyDrawdown"];
    const chart = window.echarts.getInstanceByDom(element) || window.echarts.init(element);
    chart.setOption({
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        valueFormatter: equity ? undefined : (value) => formatPercent(value, 1),
      },
      legend: { top: 8, textStyle: { color: "#9aa8c3" } },
      grid: { left: 58, right: 22, top: 48, bottom: 58 },
      xAxis: {
        type: "category",
        data: rows.map((row) => row.dataset.date),
        axisLabel: { color: "#9aa8c3", hideOverlap: true },
        axisLine: { lineStyle: { color: "#2b3655" } },
      },
      yAxis: {
        type: "value",
        axisLabel: {
          color: "#9aa8c3",
          formatter: equity ? "{value}" : (value) => `${(value * 100).toFixed(0)}%`,
        },
        splitLine: { lineStyle: { color: "#202b47" } },
      },
      series: [
        {
          name: "Original",
          type: "line",
          data: rows.map((row) => Number(row.dataset[keys[0]])),
          showSymbol: false,
          lineStyle: { color: "#63759c", width: 2 },
        },
        {
          name: "Optimized",
          type: "line",
          data: rows.map((row) => Number(row.dataset[keys[1]])),
          showSymbol: false,
          lineStyle: { color: "#67e8c9", width: 2 },
        },
        {
          name: "SPY",
          type: "line",
          data: rows.map((row) => Number(row.dataset[keys[2]])),
          showSymbol: false,
          lineStyle: { color: "#8da2d6", width: 2 },
        },
      ],
    });
    requestAnimationFrame(() => chart.resize());
  }

  function applyZscoreHeat() {
    document.querySelectorAll(".zscore-cell").forEach((cell) => {
      const value = Number(cell.dataset.zscore);
      cell.classList.remove("z-high", "z-low", "z-neutral");
      if (Number.isNaN(value)) return;
      if (value >= 1.5) cell.classList.add("z-high");
      else if (value <= -1.5) cell.classList.add("z-low");
      else cell.classList.add("z-neutral");
    });
  }

  function sortTables() {
    document.querySelectorAll("table[data-sortable] th").forEach((headerCell) => {
      if (headerCell.dataset.boundSort === "1") return;
      headerCell.dataset.boundSort = "1";
      headerCell.style.cursor = "pointer";
      headerCell.addEventListener("click", () => {
        const table = headerCell.closest("table");
        if (!table) return;
        const index = [...headerCell.parentElement.children].indexOf(headerCell);
        const bodyRows = [...table.querySelectorAll("tr")].slice(1);
        const ascending = table.dataset.sortDirection !== "asc";
        bodyRows.sort((a, b) => {
          const av = a.children[index]?.textContent?.trim() || "";
          const bv = b.children[index]?.textContent?.trim() || "";
          const an = Number(av.replace(/[%,$]/g, ""));
          const bn = Number(bv.replace(/[%,$]/g, ""));
          if (!Number.isNaN(an) && !Number.isNaN(bn)) return ascending ? an - bn : bn - an;
          return ascending ? av.localeCompare(bv) : bv.localeCompare(av);
        });
        bodyRows.forEach((row) => table.appendChild(row));
        table.dataset.sortDirection = ascending ? "asc" : "desc";
      });
    });
  }

  function bindSignalSearch() {
    document.querySelectorAll("[data-signal-search]").forEach((input) => {
      if (input.dataset.boundSignalSearch === "1") return;
      input.dataset.boundSignalSearch = "1";
      input.addEventListener("input", () => {
        const query = input.value.trim().toLowerCase();
        document.querySelectorAll(".signal-factor-list section").forEach((section) => {
          let visible = 0;
          section.querySelectorAll("a").forEach((link) => {
            const matches = !query || link.textContent.toLowerCase().includes(query);
            link.hidden = !matches;
            if (matches) visible += 1;
          });
          section.hidden = visible === 0;
        });
      });
    });
  }

  function bindSelectionFactorSearch() {
    document.querySelectorAll("[data-selection-factor-search]").forEach((input) => {
      if (input.dataset.boundSelectionSearch === "1") return;
      input.dataset.boundSelectionSearch = "1";
      input.addEventListener("input", () => {
        const query = input.value.trim().toLowerCase();
        document.querySelectorAll("[data-selection-factor-group]").forEach((group) => {
          let visible = 0;
          group.querySelectorAll("[data-selection-factor-row]").forEach((row) => {
            const matches = !query || row.textContent.toLowerCase().includes(query);
            row.hidden = !matches;
            if (matches) visible += 1;
          });
          group.hidden = visible === 0;
        });
      });
    });
  }

  // Only a ticked factor is scored, so its weight stays locked until then. That
  // keeps a stray number from looking like a selection it never was.
  function bindSelectionFactorWeights() {
    const weightInputs = () => [
      ...document.querySelectorAll(
        '[data-selection-factor-row] input[type="number"]'
      ),
    ];
    document.querySelectorAll("[data-selection-factor-row]").forEach((row) => {
      if (row.dataset.boundSelectionWeight === "1") return;
      row.dataset.boundSelectionWeight = "1";
      const toggle = row.querySelector('input[type="checkbox"]');
      const weight = row.querySelector('input[type="number"]');
      const direction = row.querySelector("select");
      if (!toggle || !weight) return;
      const apply = () => {
        weight.disabled = !toggle.checked;
        if (direction) direction.disabled = !toggle.checked;
        if (!toggle.checked) {
          weight.value = "0";
          return;
        }
        const value = Number.parseFloat(weight.value);
        if (Number.isFinite(value) && value > 0) return;
        const peers = weightInputs()
          .filter((input) => input !== weight && !input.disabled)
          .map((input) => Number.parseFloat(input.value))
          .filter((item) => Number.isFinite(item) && item > 0);
        const share = peers.length
          ? peers.reduce((total, item) => total + item, 0) / peers.length
          : 25;
        weight.value = String(Math.round(share * 100) / 100);
      };
      apply();
      toggle.addEventListener("change", () => {
        apply();
        if (toggle.checked) weight.focus();
      });
    });
  }

  function bindComboboxes() {
    document.querySelectorAll("[data-combobox]").forEach((root) => {
      if (root.dataset.boundCombobox === "1") return;
      root.dataset.boundCombobox = "1";
      const input = root.querySelector("[data-combobox-input]");
      const panel = root.querySelector("[data-combobox-panel]");
      if (!input || !panel) return;
      const options = [...panel.querySelectorAll("[data-combobox-option]")];
      const empty = panel.querySelector("[data-combobox-empty]");
      let activeIndex = -1;

      const shown = () => options.filter((option) => !option.hidden);

      function highlight(index) {
        const matches = shown();
        options.forEach((option) => option.classList.remove("active"));
        activeIndex = matches.length ? Math.max(0, Math.min(index, matches.length - 1)) : -1;
        const option = matches[activeIndex];
        if (!option) return;
        option.classList.add("active");
        option.scrollIntoView({ block: "nearest" });
      }

      function close() {
        panel.hidden = true;
        panel.scrollTop = 0;
        input.setAttribute("aria-expanded", "false");
        options.forEach((option) => option.classList.remove("active"));
        activeIndex = -1;
      }

      function filter() {
        const query = input.value.trim().toLowerCase();
        let visible = 0;
        options.forEach((option) => {
          const matches = !query || option.dataset.search.includes(query);
          option.hidden = !matches;
          if (matches) visible += 1;
        });
        if (empty) empty.hidden = visible > 0;
        panel.hidden = false;
        input.setAttribute("aria-expanded", "true");
        highlight(0);
      }

      function choose(option) {
        input.value = option.dataset.value;
        close();
        const form = input.closest("form");
        if (form) form.requestSubmit();
      }

      input.addEventListener("input", filter);
      input.addEventListener("focus", filter);
      input.addEventListener("keydown", (event) => {
        if (event.key === "Escape") {
          close();
          return;
        }
        if (event.key === "ArrowDown" || event.key === "ArrowUp") {
          event.preventDefault();
          if (panel.hidden) filter();
          else highlight(activeIndex + (event.key === "ArrowDown" ? 1 : -1));
          return;
        }
        if (event.key === "Enter" && !panel.hidden) {
          const option = shown()[activeIndex];
          if (option) {
            event.preventDefault();
            choose(option);
          }
        }
      });
      options.forEach((option) => {
        // Selection must win over the input's blur, so act before focus moves.
        option.addEventListener("mousedown", (event) => {
          event.preventDefault();
          choose(option);
        });
      });
      document.addEventListener("mousedown", (event) => {
        if (!panel.hidden && !root.contains(event.target)) close();
      });
    });
  }

  function renderOptimizerExposure() {
    const element = document.querySelector('[data-chart="optimizer-exposure"]');
    const rows = [...document.querySelectorAll("[data-optimizer-exposure]")];
    if (!element || !window.echarts || !rows.length) return;
    const chart = window.echarts.getInstanceByDom(element) || window.echarts.init(element);
    chart.setOption({
      backgroundColor: "transparent",
      title: { text: element.dataset.title || "Before / after factor exposure", textStyle: { color: "#e8edf7", fontSize: 14 } },
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" } },
      legend: { top: 28, textStyle: { color: "#9aa8c3" } },
      grid: { left: 52, right: 20, top: 72, bottom: 88 },
      xAxis: {
        type: "category",
        data: rows.map((row) => row.dataset.name),
        axisLabel: { color: "#9aa8c3", rotate: 38 },
        axisLine: { lineStyle: { color: "#2b3655" } },
      },
      yAxis: { type: "value", axisLabel: { color: "#9aa8c3" }, splitLine: { lineStyle: { color: "#202b47" } } },
      series: [
        { name: "Before", type: "bar", data: rows.map((row) => Number(row.dataset.original)), itemStyle: { color: "#63759c" } },
        { name: "After", type: "bar", data: rows.map((row) => Number(row.dataset.optimized)), itemStyle: { color: "#67e8c9" } },
      ],
    });
    requestAnimationFrame(() => chart.resize());
  }

  function formatPercent(value, digits = 2) {
    const number = Number(value);
    return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "—";
  }

  function renderFactorDialog(dialog) {
    const payload = dialog.factorPayload;
    const frequency = dialog.querySelector("[data-factor-frequency]")?.value || "monthly";
    const series = payload?.series?.[frequency] || [];
    if (!payload || !series.length) return;

    const factor = payload.factor;
    const construction = factor.construction || {};
    const frequencyLabel = `${frequency[0].toUpperCase()}${frequency.slice(1)}`;
    dialog.querySelector("[data-factor-dialog-name]").textContent = factor.name;
    dialog.querySelector("[data-factor-dialog-subtitle]").textContent =
      `${frequencyLabel} returns through ${payload.build.as_of}.`;
    dialog.querySelector("[data-factor-description]").textContent = factor.description;
    dialog.querySelector("[data-factor-kind]").textContent =
      (construction.kind || "unknown").replaceAll("_", " ");
    dialog.querySelector("[data-factor-period-title]").textContent = `${frequencyLabel} return`;

    const legs = construction.inverted
      ? `Inverse ${construction.long_leg}`
      : [construction.long_leg && `Long ${construction.long_leg}`, construction.short_leg && `Short ${construction.short_leg}`]
        .filter(Boolean)
        .join(" / ");
    const legsRow = dialog.querySelector("[data-factor-legs-row]");
    legsRow.hidden = !legs;
    dialog.querySelector("[data-factor-legs]").textContent = legs || "—";

    const purification = (construction.purified_against || []).join(", ");
    const purificationRow = dialog.querySelector("[data-factor-purification-row]");
    purificationRow.hidden = !purification;
    dialog.querySelector("[data-factor-purification]").textContent = purification || "—";

    const estimation = [construction.estimation_frequency, construction.rolling_window]
      .filter(Boolean)
      .join("; ");
    const estimationRow = dialog.querySelector("[data-factor-estimation-row]");
    estimationRow.hidden = !estimation;
    dialog.querySelector("[data-factor-estimation]").textContent = estimation || "—";

    const target = Number(construction.volatility_target);
    const volatilityRow = dialog.querySelector("[data-factor-volatility-row]");
    volatilityRow.hidden = !Number.isFinite(target) || target <= 0;
    dialog.querySelector("[data-factor-volatility]").textContent =
      Number.isFinite(target) && target > 0 ? formatPercent(target, 0) + " annualized" : "—";

    const basket = construction.basket || [];
    const basketRow = dialog.querySelector("[data-factor-basket-row]");
    basketRow.hidden = !basket.length;
    dialog.querySelector("[data-factor-basket]").textContent = basket.join(", ");

    if (!window.echarts) return;
    const text = "#9aa8c3";
    const line = "#2b3655";
    const dates = series.map((item) => item.date);
    const cumulativeElement = dialog.querySelector('[data-chart="factor-dialog-cumulative"]');
    const cumulativeChart = window.echarts.getInstanceByDom(cumulativeElement)
      || window.echarts.init(cumulativeElement);
    cumulativeChart.setOption({
      backgroundColor: "transparent",
      tooltip: { trigger: "axis" },
      grid: { left: 58, right: 18, top: 24, bottom: 58 },
      xAxis: {
        type: "category",
        data: dates,
        axisLabel: { color: text, hideOverlap: true },
        axisLine: { lineStyle: { color: line } },
      },
      yAxis: {
        type: "value",
        scale: true,
        axisLabel: { color: text },
        splitLine: { lineStyle: { color: line } },
      },
      series: [{
        name: "Cumulative index",
        type: "line",
        data: series.map((item) => item.cumulative_index),
        showSymbol: false,
        lineStyle: { color: "#67e8c9", width: 2 },
      }],
    }, true);

    const periodElement = dialog.querySelector('[data-chart="factor-dialog-period"]');
    const periodChart = window.echarts.getInstanceByDom(periodElement)
      || window.echarts.init(periodElement);
    periodChart.setOption({
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        valueFormatter: (value) => `${Number(value).toFixed(2)}%`,
      },
      grid: { left: 58, right: 18, top: 24, bottom: 58 },
      xAxis: {
        type: "category",
        data: dates,
        axisLabel: { color: text, hideOverlap: true },
        axisLine: { lineStyle: { color: line } },
      },
      yAxis: {
        type: "value",
        axisLabel: { color: text, formatter: "{value}%" },
        splitLine: { lineStyle: { color: line } },
      },
      series: [{
        name: `${frequencyLabel} return`,
        type: "bar",
        data: series.map((item) => {
          const value = Number(item.return) * 100;
          return { value, itemStyle: { color: value < 0 ? "#d8798c" : "#67e8c9" } };
        }),
      }],
    }, true);
    requestAnimationFrame(() => {
      cumulativeChart.resize();
      periodChart.resize();
    });
  }

  function bindFactorDialog() {
    const dialog = document.querySelector("[data-factor-dialog]");
    if (!dialog || dialog.dataset.boundFactorDialog === "1") return;
    dialog.dataset.boundFactorDialog = "1";
    const frequency = dialog.querySelector("[data-factor-frequency]");
    frequency.addEventListener("change", () => renderFactorDialog(dialog));
    dialog.querySelector("[data-factor-dialog-close]").addEventListener("click", () => dialog.close());
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog) dialog.close();
    });
    document.querySelectorAll("[data-factor-detail-url]").forEach((button) => {
      button.addEventListener("click", async () => {
        frequency.value = "monthly";
        dialog.querySelector("[data-factor-dialog-name]").textContent = button.dataset.factorName;
        dialog.querySelector("[data-factor-dialog-loading]").hidden = false;
        dialog.querySelector("[data-factor-dialog-error]").hidden = true;
        dialog.querySelector("[data-factor-dialog-content]").hidden = true;
        dialog.showModal();
        try {
          const response = await fetch(button.dataset.factorDetailUrl, {
            headers: { Accept: "application/json" },
          });
          const payload = await response.json();
          if (!response.ok) throw new Error(payload.error || `Factor request failed (${response.status}).`);
          dialog.factorPayload = payload;
          dialog.querySelector("[data-factor-dialog-loading]").hidden = true;
          dialog.querySelector("[data-factor-dialog-content]").hidden = false;
          renderFactorDialog(dialog);
        } catch (error) {
          dialog.querySelector("[data-factor-dialog-loading]").hidden = true;
          const notice = dialog.querySelector("[data-factor-dialog-error]");
          notice.textContent = error.message;
          notice.hidden = false;
        }
      });
    });
  }

  function bindOptimizerControls() {
    const bindRangeOutput = (input) => {
      if (input.dataset.boundRangeOutput === "1") return;
      input.dataset.boundRangeOutput = "1";
      const output = input.parentElement?.querySelector("output");
      const update = () => {
        if (output) output.value = input.value;
      };
      input.addEventListener("input", update);
      update();
    };

    document.querySelectorAll("[data-weight-constraint-mode]").forEach((select) => {
      if (select.dataset.boundWeightMode === "1") return;
      select.dataset.boundWeightMode = "1";
      const root = select.closest("[data-weight-constraints]");
      const update = () => {
        const relative = select.value === "relative";
        const minLabel = root?.querySelector("[data-weight-min-label]");
        const maxLabel = root?.querySelector("[data-weight-max-label]");
        const capLabel = root?.querySelector("[data-weight-cap-label]");
        if (minLabel) minLabel.firstChild.textContent = relative ? "Decrease by (%) " : "Minimum weight (%) ";
        if (maxLabel) maxLabel.firstChild.textContent = relative ? "Increase by (%) " : "Maximum weight (%) ";
        if (capLabel) {
          capLabel.hidden = !relative;
          const input = capLabel.querySelector("input");
          if (input) input.disabled = !relative;
        }
      };
      select.addEventListener("change", () => {
        const minInput = root?.querySelector('input[name="min_weight"]');
        const maxInput = root?.querySelector('input[name="max_weight"]');
        if (select.value === "relative" && minInput?.value === "0" && maxInput?.value === "15") {
          minInput.value = "5";
          maxInput.value = "5";
        } else if (select.value === "absolute" && minInput?.value === "5" && maxInput?.value === "5") {
          minInput.value = "0";
          maxInput.value = "15";
        }
        update();
      });
      update();
    });

    document.querySelectorAll('.optimizer-form select[name="objective"]').forEach((select) => {
      if (select.dataset.boundObjective === "1") return;
      select.dataset.boundObjective = "1";
      const update = () => {
        const form = select.closest("form");
        const riskField = form?.querySelector("[data-risk-aversion-field]");
        if (riskField) riskField.hidden = select.value !== "mean_variance";
        const expectedModel = form?.querySelector("[data-expected-return-model]");
        if (expectedModel && select.value !== "min_variance" && !expectedModel.value) {
          expectedModel.value = "factor_premium";
          expectedModel.dispatchEvent(new Event("change"));
        }
      };
      select.addEventListener("change", update);
      update();
    });

    const updateEmptyMessage = (container) => {
      const message = container?.querySelector("[data-empty-message]");
      if (message) message.hidden = container.children.length > 1;
    };
    const attachRemove = (row, option, container) => {
      row.querySelector("[data-remove-row]")?.addEventListener("click", () => {
        if (option) option.disabled = false;
        row.remove();
        updateEmptyMessage(container);
      });
    };

    document.querySelectorAll("[data-add-premium]").forEach((button) => {
      if (button.dataset.boundAdd === "1") return;
      button.dataset.boundAdd = "1";
      const select = button.parentElement.querySelector("[data-add-premium-select]");
      const container = button.closest("form")?.querySelector("[data-factor-premium-rows]");
      button.addEventListener("click", () => {
        const option = select?.selectedOptions[0];
        if (!option?.value || !container) return;
        const row = document.createElement("div");
        row.className = "optimizer-premium-row";
        row.dataset.constraintKey = option.value;
        row.innerHTML = `<strong data-row-name></strong>
          <label>Prior premium (%) <input type="number" name="factor_premium_${option.value}" step=".1" value="0"></label>
          <button type="button" class="danger-link" data-remove-row aria-label="Remove factor premium">Remove</button>`;
        row.querySelector("[data-row-name]").textContent = option.dataset.name || option.textContent;
        container.appendChild(row);
        option.disabled = true;
        select.value = "";
        attachRemove(row, option, container);
        updateEmptyMessage(container);
      });
      container?.querySelectorAll("[data-constraint-key]").forEach((row) => {
        const option = select?.querySelector(`option[value="${row.dataset.constraintKey}"]`);
        attachRemove(row, option, container);
      });
    });

    document.querySelectorAll("[data-add-factor]").forEach((button) => {
      if (button.dataset.boundAdd === "1") return;
      button.dataset.boundAdd = "1";
      const select = button.parentElement.querySelector("[data-add-factor-select]");
      const container = button.closest("section")?.querySelector("[data-factor-constraint-rows]");
      button.addEventListener("click", () => {
        const option = select?.selectedOptions[0];
        if (!option?.value || !container) return;
        const row = document.createElement("div");
        row.className = "optimizer-factor-constraint-row";
        row.dataset.constraintKey = option.value;
        row.innerHTML = `<div><strong data-row-name></strong><small data-row-family></small></div>
          <label>Absolute min<input type="number" name="factor_min_${option.value}" step=".01"></label>
          <label>Absolute max<input type="number" name="factor_max_${option.value}" step=".01"></label>
          <label>SPY-relative min<input type="number" name="relative_min_${option.value}" step=".01"></label>
          <label>SPY-relative max<input type="number" name="relative_max_${option.value}" step=".01"></label>
          <label>Component cap<input type="number" name="component_cap_${option.value}" min="0" step=".001"></label>
          <button type="button" class="danger-link" data-remove-row aria-label="Remove factor constraint">Remove</button>`;
        row.querySelector("[data-row-name]").textContent = option.dataset.name || option.textContent;
        row.querySelector("[data-row-family]").textContent = option.dataset.family || "";
        container.appendChild(row);
        option.disabled = true;
        select.value = "";
        attachRemove(row, option, container);
        updateEmptyMessage(container);
      });
    });

    document.querySelectorAll("[data-add-sector]").forEach((button) => {
      if (button.dataset.boundAdd === "1") return;
      button.dataset.boundAdd = "1";
      const select = button.parentElement.querySelector("[data-add-sector-select]");
      const container = button.closest("section")?.querySelector("[data-sector-constraint-rows]");
      button.addEventListener("click", () => {
        const option = select?.selectedOptions[0];
        if (!option?.value || !container) return;
        const row = document.createElement("div");
        row.className = "optimizer-sector-row";
        row.dataset.constraintKey = option.value;
        row.innerHTML = `<input type="hidden" name="sector_name_${option.value}">
          <strong data-row-name></strong>
          <label>Minimum (%)<input type="number" name="sector_min_${option.value}" min="0" max="100" step=".1"></label>
          <label>Maximum (%)<input type="number" name="sector_max_${option.value}" min="0" max="100" step=".1"></label>
          <button type="button" class="danger-link" data-remove-row aria-label="Remove sector constraint">Remove</button>`;
        row.querySelector('input[type="hidden"]').value = option.dataset.name || option.textContent;
        row.querySelector("[data-row-name]").textContent = option.dataset.name || option.textContent;
        container.appendChild(row);
        option.disabled = true;
        select.value = "";
        attachRemove(row, option, container);
        updateEmptyMessage(container);
      });
    });

    document.querySelectorAll("[data-optimizer-factor-search]").forEach((input) => {
      if (input.dataset.boundOptimizerSearch === "1") return;
      input.dataset.boundOptimizerSearch = "1";
      input.addEventListener("input", () => {
        const query = input.value.trim().toLowerCase();
        document.querySelectorAll("[data-optimizer-factor-group]").forEach((group) => {
          let visible = 0;
          group.querySelectorAll("[data-optimizer-factor-row]").forEach((row) => {
            const matches = !query || row.textContent.toLowerCase().includes(query);
            row.hidden = !matches;
            if (matches) visible += 1;
          });
          group.hidden = visible === 0;
        });
      });
    });
    document.querySelectorAll("[data-range-output]").forEach(bindRangeOutput);
    document.querySelectorAll("[data-expected-return-model]").forEach((select) => {
      if (select.dataset.boundExpectedReturn === "1") return;
      select.dataset.boundExpectedReturn = "1";
      const panel = select.closest("form")?.querySelector("[data-expected-returns-panel]");
      const premiumPanel = select.closest("form")?.querySelector("[data-factor-premium-panel]");
      const equalSharpePanel = select.closest("form")?.querySelector("[data-equal-sharpe-panel]");
      // A hidden panel still submits and still blocks HTML5 validation, and the
      // browser cannot show a bubble on a field it cannot focus.
      const setPanel = (element, active) => {
        if (!element) return;
        element.hidden = !active;
        element
          .querySelectorAll("input, select, textarea")
          .forEach((field) => (field.disabled = !active));
      };
      const update = () => {
        setPanel(premiumPanel, select.value === "factor_premium");
        setPanel(equalSharpePanel, select.value === "equal_sharpe");
        if (panel && select.value === "user_supplied") panel.open = true;
      };
      select.addEventListener("change", update);
      update();
    });
    document.querySelectorAll(
      '.optimizer-form select[name="portfolio_id"], .optimizer-form select[name="factor_build_id"], .optimizer-form select[name="model_level"]',
    ).forEach((select) => {
      if (select.dataset.boundOptimizerContext === "1") return;
      select.dataset.boundOptimizerContext = "1";
      select.addEventListener("change", () => {
        const form = select.closest("form");
        if (!form) return;
        const url = new URL(window.location.href);
        url.searchParams.delete("mode");
        ["portfolio_id", "factor_build_id", "model_level"].forEach((name) => {
          const value = form.querySelector(`[name="${name}"]`)?.value;
          if (value) url.searchParams.set(name, value);
          else url.searchParams.delete(name);
        });
        window.location.assign(url);
      });
    });
  }

  function resizeChartsWithin(root) {
    if (!window.echarts) return;
    root.querySelectorAll(".chart").forEach((element) => {
      const chart = window.echarts.getInstanceByDom(element);
      if (chart) chart.resize();
    });
  }

  // Collapsed panels keep their charts at zero size, so a stored open/closed
  // state has to survive the full page reload that variant links trigger.
  function bindCollapsiblePanels() {
    const storageKey = (key) => `quantlab.collapse.${key}`;

    document.querySelectorAll(".collapse-panel > details[data-collapse-key]").forEach((details) => {
      if (details.dataset.boundCollapse === "1") return;
      details.dataset.boundCollapse = "1";
      const key = details.dataset.collapseKey;
      if (details.hasAttribute("data-collapse-force-open")) {
        details.open = true;
      } else {
        let stored = null;
        try {
          stored = window.localStorage.getItem(storageKey(key));
        } catch (error) {
          stored = null;
        }
        if (stored === "open") details.open = true;
        else if (stored === "closed") details.open = false;
      }
      details.addEventListener("toggle", () => {
        if (!details.hasAttribute("data-collapse-force-open")) {
          try {
            window.localStorage.setItem(storageKey(key), details.open ? "open" : "closed");
          } catch (error) {
            /* storage is unavailable; the panel still toggles */
          }
        }
        if (details.open) requestAnimationFrame(() => resizeChartsWithin(details));
      });
    });
  }

  // Charts nested inside a plain details also start at zero size, so they need
  // the same resize on open as the keyed collapse panels.
  function bindNestedChartDetails() {
    document.querySelectorAll("details:not([data-collapse-key])").forEach((details) => {
      if (details.dataset.boundChartToggle === "1") return;
      if (!details.querySelector(".chart")) return;
      details.dataset.boundChartToggle = "1";
      details.addEventListener("toggle", () => {
        if (details.open) requestAnimationFrame(() => resizeChartsWithin(details));
      });
    });
  }

  function hydrate() {
    const exposureRows = [...document.querySelectorAll("[data-stock-factor]")];
    if (exposureRows.length) {
      renderChart(
        document.querySelector('[data-chart="stock-exposure"]'),
        exposureRows,
        "Base + Sector factor betas",
        "bar",
      );
    } else if (document.querySelector('[data-chart="stock-exposure"]')) {
      renderChart(
        document.querySelector('[data-chart="stock-exposure"]'),
        [...document.querySelectorAll("[data-stock-history]")].reverse(),
        "Historical adjusted R²",
      );
    }
    renderChart(
      document.querySelector('[data-chart="stock-r2-history"]'),
      [...document.querySelectorAll("[data-stock-r2-history]")],
      "Adjusted R² by month-end",
    );
    renderChart(
      document.querySelector('[data-chart="stock-beta-history"]'),
      [...document.querySelectorAll("[data-stock-beta-history]")],
      "Factor beta by month-end",
    );
    renderChart(
      document.querySelector('[data-chart="stock-contribution"]'),
      [...document.querySelectorAll("[data-stock-contribution]")],
      "Latest factor contribution",
      "bar",
    );
    renderChart(
      document.querySelector('[data-chart="signal-ranking"]'),
      [...document.querySelectorAll("[data-signal-ranking]")],
      "Factor exposure z-score",
      "bar",
    );
    renderChart(
      document.querySelector('[data-chart="signal-beta-history"]'),
      [...document.querySelectorAll("[data-signal-beta-history]")],
      "Factor beta history",
    );
    renderChart(
      document.querySelector('[data-chart="signal-percentile-history"]'),
      [...document.querySelectorAll("[data-signal-percentile-history]")],
      "Cross-sectional percentile",
    );
    renderChart(
      document.querySelector('[data-chart="selection-score"]'),
      [...document.querySelectorAll("[data-selection-score]")],
      "Weighted factor score",
      "bar",
    );
    renderChart(
      document.querySelector('[data-chart="portfolio-exposure"]'),
      [...document.querySelectorAll("[data-portfolio-exposure]")],
      "Weighted portfolio exposures",
      "bar",
    );
    renderChart(
      document.querySelector('[data-chart="portfolio-contribution"]'),
      [...document.querySelectorAll("[data-portfolio-contribution]")],
      "Stored component risk",
      "bar",
    );
    renderHistoryChart("equity");
    renderHistoryChart("drawdown");
    renderOptimizerHistoryChart("equity");
    renderOptimizerHistoryChart("drawdown");
    renderOptimizerExposure();
    applyZscoreHeat();
    sortTables();
    bindSignalSearch();
    bindSelectionFactorSearch();
    bindSelectionFactorWeights();
    bindComboboxes();
    bindOptimizerControls();
    bindFactorDialog();
    bindCollapsiblePanels();
    bindNestedChartDetails();
  }

  window.addEventListener("load", hydrate);
  document.body.addEventListener("htmx:afterSwap", hydrate);
  // htmx halts a request on HTML5 validation failure without reporting it, which
  // looks like a dead button when the offending field is scrolled out of view.
  document.body.addEventListener("htmx:validation:halted", (event) => {
    const form = event.target.closest("form") || event.target;
    const invalid = form.querySelector?.(":invalid");
    if (!invalid) return;
    // A control inside a closed <details> cannot be focused, so reportValidity
    // would silently do nothing.
    for (let node = invalid.closest("details"); node; node = node.parentElement?.closest("details")) {
      node.open = true;
    }
    const label = invalid.closest("label")?.textContent.trim().split("\n")[0];
    const target = form.getAttribute("hx-target");
    const banner = target && document.querySelector(target);
    if (banner) {
      banner.innerHTML = "";
      const notice = document.createElement("div");
      notice.className = "warning";
      notice.textContent = `${label || invalid.name || "A required field"} is required.`;
      banner.appendChild(notice);
    }
    invalid.scrollIntoView({ block: "center", behavior: "smooth" });
    invalid.reportValidity();
  });
  window.addEventListener("resize", () => {
    document.querySelectorAll(".chart").forEach((element) => {
      const chart = window.echarts && window.echarts.getInstanceByDom(element);
      if (chart) chart.resize();
    });
  });
  if (document.readyState === "complete") hydrate();
})();
