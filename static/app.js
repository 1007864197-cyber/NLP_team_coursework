/* The browser owns presentation only. All restaurant answers come from the Python API. */
(() => {
  "use strict";

  const STORAGE_KEY = "olive-table-conversation-v1";
  const MAX_MESSAGES = 60;
  const INITIAL_PROMPTS = ["What would you recommend?", "Do you have vegan dishes?", "What are your opening hours?"];
  const $ = (id) => document.getElementById(id);
  const ui = {
    form: $("chat-form"), input: $("message"), send: $("send-button"),
    conversation: $("conversation"), prompts: $("suggested-prompts"),
    reset: $("new-chat"), export: $("export-chat"), count: $("character-count"),
    error: $("chat-error"), errorMessage: $("error-message"), retry: $("config-retry"),
    status: $("service-status"), dot: $("status-dot"), mode: $("answer-mode"),
    menu: $("menu-list"), menuCount: $("menu-count"), announcement: $("announcement"),
  };
  const state = {config: null, sessionId: null, messages: [], mode: "enhanced", busy: false, ready: false, filter: "all", category: "all", search: "", selectedDish: null, prompts: INITIAL_PROMPTS};

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = String(text);
    return node;
  }

  function announce(text) {
    ui.announcement.textContent = "";
    // A new text node lets screen readers announce repeated notices as well.
    requestAnimationFrame(() => { ui.announcement.textContent = text; });
  }

  function setError(message, configError = false) {
    ui.errorMessage.textContent = message;
    ui.error.hidden = !message;
    ui.retry.hidden = !configError;
  }

  function updateControls() {
    ui.input.disabled = !state.ready || state.busy;
    ui.send.disabled = !state.ready || state.busy || !ui.input.value.trim();
    ui.reset.disabled = !state.ready || state.busy;
    ui.export.disabled = !state.ready || state.busy || state.messages.length === 0;
    ui.mode.disabled = state.busy;
    document.querySelectorAll(".prompt-button, .dish-ask").forEach((button) => {
      button.disabled = !state.ready || state.busy;
    });
    ui.count.textContent = `${ui.input.value.length} / 1200`;
  }

  function sizeInput() {
    ui.input.style.height = "auto";
    ui.input.style.height = `${Math.min(ui.input.scrollHeight, 115)}px`;
    updateControls();
  }

  function saveSession() {
    try {
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify({
        sessionId: state.sessionId, messages: state.messages.slice(-MAX_MESSAGES),
        mode: state.mode, prompts: state.prompts,
      }));
    } catch (_) { /* Chat still works when private browsing blocks storage. */ }
  }

  function restoreSession() {
    try {
      const saved = JSON.parse(sessionStorage.getItem(STORAGE_KEY) || "null");
      if (!saved || !Array.isArray(saved.messages)) return;
      state.sessionId = typeof saved.sessionId === "string" ? saved.sessionId : null;
      state.mode = saved.mode === "baseline" ? "baseline" : "enhanced";
      // Stored text is treated as data and is never inserted as HTML.
      state.messages = saved.messages.filter((message) => message && ["user", "assistant", "system"].includes(message.role) && typeof message.text === "string")
        .slice(-MAX_MESSAGES).map((message) => ({...message, text: message.text.slice(0, 20000)}));
      state.prompts = Array.isArray(saved.prompts) ? saved.prompts.filter((p) => typeof p === "string").slice(0, 3) : INITIAL_PROMPTS;
      ui.mode.value = state.mode;
    } catch (_) { /* Invalid or old storage starts an empty conversation. */ }
  }

  function scrollToLatest() {
    requestAnimationFrame(() => { ui.conversation.scrollTop = ui.conversation.scrollHeight; });
  }

  function welcomeMessage() {
    const node = element("article", "message assistant-message welcome-message");
    node.append(element("p", "message-label", "OLIVE"));
    const content = element("div", "message-content");
    content.append(element("span", "welcome-title", "Hello, pull up a chair."));
    content.append(document.createTextNode("I’m Olive, your guide to our kitchen. Tell me what you’re in the mood for, or ask about the menu, opening hours, and planning your visit."));
    content.append(element("span", "welcome-signature", "What sounds good to you?"));
    node.append(content);
    return node;
  }

  function evidenceDetails(metadata) {
    if (!metadata || typeof metadata !== "object") return null;
    const details = element("details", "answer-evidence");
    details.append(element("summary", "", "How this answer was found"));
    const body = element("div", "evidence-content");
    if (typeof metadata.method === "string") body.append(element("p", "", `Method: ${metadata.method}`));
    if (typeof metadata.intent === "string") body.append(element("p", "", `Intent: ${metadata.intent.replace(/_/g, " ")}`));
    const score = Number(metadata.confidence);
    if (Number.isFinite(score)) body.append(element("p", "", `Model score: ${score.toFixed(3)} (classification, similarity or rule score; not an accuracy estimate)`));
    const sources = Array.isArray(metadata.sources) ? metadata.sources : [];
    const titles = sources.filter((source) => source && typeof source.title === "string").map((source) => source.title);
    body.append(element("p", "", titles.length ? `Information: ${titles.join(" · ")}` : "Information: conversational response"));
    if (Number.isFinite(Number(metadata.elapsed_ms))) body.append(element("p", "", `Response time: ${Math.round(Number(metadata.elapsed_ms))} ms`));
    if (metadata.context && typeof metadata.context === "object") {
      const context = metadata.context;
      const preferences = [];
      if (context.diet) preferences.push(String(context.diet));
      if (Array.isArray(context.excluded_ingredients) && context.excluded_ingredients.length) preferences.push(`avoiding ${context.excluded_ingredients.join(", ")}`);
      if (context.max_price !== null && context.max_price !== undefined) preferences.push(`budget ${money(context.max_price)}`);
      if (context.spice) preferences.push(String(context.spice));
      if (preferences.length) body.append(element("p", "", `Remembered preferences: ${preferences.join(" · ")}`));
    }
    details.append(body);
    return details;
  }

  function messageNode(message) {
    if (message.role === "system") return element("p", "message system-message", message.text);
    const isUser = message.role === "user";
    const node = element("article", `message ${isUser ? "user-message" : "assistant-message"}`);
    node.append(element("p", "message-label", isUser ? "YOU" : "OLIVE"));
    node.append(element("div", "message-content", message.text));
    if (!isUser) {
      const evidence = evidenceDetails(message.metadata);
      if (evidence) node.append(evidence);
    }
    return node;
  }

  function renderConversation() {
    ui.conversation.replaceChildren(welcomeMessage());
    state.messages.forEach((message) => ui.conversation.append(messageNode(message)));
    scrollToLatest();
  }

  function typingNode() {
    const node = element("div", "message assistant-message typing-message");
    node.setAttribute("aria-label", "Olive is preparing a reply");
    const content = element("div", "message-content");
    for (let i = 0; i < 3; i += 1) content.append(element("span"));
    node.append(content);
    return node;
  }

  function renderPrompts() {
    ui.prompts.replaceChildren();
    state.prompts.slice(0, 3).forEach((prompt) => {
      const button = element("button", "prompt-button", prompt);
      button.type = "button";
      button.disabled = state.busy || !state.ready;
      button.addEventListener("click", () => sendPrompt(prompt));
      ui.prompts.append(button);
    });
  }

  function money(value) {
    const currency = state.config?.restaurant?.currency || "GBP";
    const amount = Number(value);
    if (!Number.isFinite(amount)) return "Ask us";
    try { return new Intl.NumberFormat("en-GB", {style: "currency", currency}).format(amount); }
    catch (_) { return `£${amount.toFixed(2)}`; }
  }

  function normalizedTags(dish) {
    return (Array.isArray(dish.tags) ? dish.tags : []).map((tag) => String(tag).toLowerCase().replace(/_/g, "-"));
  }

  function renderMenu() {
    const allDishes = Array.isArray(state.config?.menu) ? state.config.menu : [];
    const dishes = allDishes.filter((dish) => {
      const tags = normalizedTags(dish);
      const dietMatches = state.filter === "all" || tags.includes(state.filter) || (state.filter === "vegetarian" && tags.includes("vegan"));
      const categoryMatches = state.category === "all" || dish.category === state.category;
      const searchable = [dish.name, dish.description, ...(Array.isArray(dish.ingredients) ? dish.ingredients : [])].join(" ").toLowerCase();
      return dietMatches && categoryMatches && (!state.search || searchable.includes(state.search));
    });
    ui.menuCount.textContent = `${dishes.length} ${dishes.length === 1 ? "ITEM" : "ITEMS"}`;
    ui.menu.replaceChildren();
    if (!dishes.length) {
      ui.menu.append(element("p", "menu-empty", "Nothing on the menu matches just yet. Try another category, dietary filter, or search."));
      const clear = element("button", "prompt-button", "Clear menu filters");
      clear.type = "button";
      clear.addEventListener("click", clearMenuFilters);
      ui.menu.append(clear);
      return;
    }
    dishes.forEach((dish) => {
      const card = element("article", "menu-item");
      card.append(element("h3", "dish-name", dish.name || "Menu item"));
      card.append(element("p", "dish-price", money(dish.price)));
      card.append(element("p", "dish-description", dish.description || "Ask Olive for the details."));
      const tags = element("div", "dish-tags");
      if (dish.category) tags.append(element("span", "dish-tag category-tag", dish.category));
      // Vegan dishes need only one dietary badge rather than overlapping labels.
      const dietaryTags = normalizedTags(dish).filter((tag) => ["vegan", "vegetarian", "gluten-free"].includes(tag));
      dietaryTags.filter((tag) => !(tag === "vegetarian" && dietaryTags.includes("vegan"))).forEach((tag) => tags.append(element("span", "dish-tag", tag)));
      card.append(tags);
      const detail = element("button", "dish-detail-trigger", "View dish ↗");
      detail.type = "button";
      detail.setAttribute("aria-label", `View details for ${dish.name || "this dish"}`);
      detail.addEventListener("click", () => openDish(dish));
      card.append(detail);
      ui.menu.append(card);
    });
    ui.menu.scrollTop = 0;
  }

  function renderCategories() {
    const categories = [...new Set((state.config?.menu || []).map((dish) => dish.category).filter(Boolean))];
    const names = {starter: "Starters", main: "Main dishes", side: "Sides", dessert: "Desserts", drink: "Drinks"};
    const select = $("menu-category");
    select.replaceChildren();
    const all = element("option", "", "All categories");
    all.value = "all";
    select.append(all);
    categories.forEach((category) => {
      const option = element("option", "", names[category] || String(category));
      option.value = category;
      select.append(option);
    });
    select.value = state.category;
  }

  function clearMenuFilters() {
    state.filter = "all";
    state.category = "all";
    state.search = "";
    $("menu-category").value = "all";
    $("menu-search").value = "";
    document.querySelectorAll(".filter-button").forEach((button) => {
      const selected = button.dataset.filter === "all";
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
    });
    renderMenu();
    announce("Menu filters have been cleared.");
  }

  function openDish(dish) {
    state.selectedDish = dish;
    $("dish-dialog-title").textContent = dish.name || "From our kitchen";
    $("dish-dialog-price").textContent = money(dish.price);
    $("dish-dialog-category").textContent = String(dish.category || "ON THE MENU").toUpperCase();
    $("dish-dialog-description").textContent = dish.description || "Ask Olive for more about this dish.";
    $("dish-dialog-ingredients").textContent = Array.isArray(dish.ingredients) && dish.ingredients.length ? dish.ingredients.join(", ") : "Please ask the restaurant team for the full ingredient list.";
    $("dish-dialog-allergens").textContent = Array.isArray(dish.allergens) && dish.allergens.length ? `Listed allergens: ${dish.allergens.join(", ")}.` : "No allergens are listed for this item. This is not a guarantee that it is allergen-free.";
    const tags = $("dish-dialog-tags");
    tags.replaceChildren();
    normalizedTags(dish).forEach((tag) => tags.append(element("span", "dish-tag", tag)));
    $("dish-dialog-ask").disabled = state.busy || !state.ready;
    $("dish-dialog").showModal();
  }

  function hoursEntries(hours) {
    if (typeof hours === "string") return [["", hours]];
    if (Array.isArray(hours)) return hours.map((entry) => typeof entry === "string" ? ["", entry] : [entry.day || entry.days || "", entry.hours || entry.time || ""]);
    if (hours && typeof hours === "object") return Object.entries(hours).map(([key, value]) => [key, String(value)]);
    return [["", "Ask Olive about opening hours."]];
  }

  function renderRestaurant() {
    const restaurant = state.config.restaurant || {};
    const hours = hoursEntries(restaurant.hours);
    $("opening-summary").textContent = hours.length ? hours[0].filter(Boolean).join(" · ") : "Ask Olive for opening hours";
    $("address-summary").textContent = typeof restaurant.address === "string" ? restaurant.address : "Ask Olive for directions";
    const fullHours = $("full-hours");
    fullHours.replaceChildren();
    hours.forEach(([day, time]) => {
      const row = element("p", "hours-row");
      if (day) row.append(element("span", "", day));
      row.append(element("span", "", time));
      fullHours.append(row);
    });
    const contact = $("contact-details");
    contact.replaceChildren();
    if (restaurant.phone) {
      const phone = element("a", "", restaurant.phone);
      phone.href = `tel:${String(restaurant.phone).replace(/[^+0-9]/g, "")}`;
      contact.append(phone, document.createElement("br"));
    }
    if (restaurant.email) {
      const email = element("a", "", restaurant.email);
      email.href = `mailto:${restaurant.email}`;
      contact.append(email);
    }
    if (!restaurant.email && !restaurant.phone) contact.textContent = "Ask Olive how to get in touch.";
  }

  async function loadConfig() {
    state.ready = false;
    setError("");
    ui.status.textContent = "Getting ready…";
    ui.dot.className = "status-dot";
    updateControls();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 12000);
    try {
      const response = await fetch("/api/config", {signal: controller.signal, headers: {Accept: "application/json"}});
      if (!response.ok) throw new Error("Restaurant information could not be loaded.");
      const config = await response.json();
      if (!config || !config.restaurant || !Array.isArray(config.menu)) throw new Error("Restaurant information is incomplete.");
      state.config = config;
      state.ready = true;
      if (!state.messages.length && Array.isArray(config.quick_prompts)) state.prompts = config.quick_prompts.filter((p) => typeof p === "string").slice(0, 3);
      renderRestaurant();
      renderCategories();
      renderMenu();
      renderPrompts();
      ui.status.textContent = "Your restaurant concierge";
      ui.dot.className = "status-dot ready";
      updateControls();
    } catch (_) {
      ui.status.textContent = "Temporarily unavailable";
      ui.dot.className = "status-dot error";
      $("opening-summary").textContent = "Details unavailable";
      $("address-summary").textContent = "Please try again";
      ui.menu.replaceChildren(element("p", "menu-empty", "We couldn’t load the menu. Please try again."));
      setError("Olive is not connected yet. Check that the Python application is running, then try again.", true);
    } finally {
      clearTimeout(timeout);
      updateControls();
    }
  }

  function sendPrompt(prompt) {
    if (!state.ready || state.busy) return;
    ui.input.value = prompt;
    sizeInput();
    ui.form.requestSubmit();
    if (window.innerWidth < 851) $("chat-title").scrollIntoView({behavior: "smooth", block: "start"});
  }

  async function sendMessage(event) {
    event.preventDefault();
    const text = ui.input.value.trim();
    if (!text || state.busy || !state.ready) return;
    if (text.length > 1200) {
      setError("Please keep your message to 1,200 characters or fewer.");
      return;
    }
    state.busy = true;
    setError("");
    updateControls();
    const userMessage = {role: "user", text};
    const pendingUserNode = messageNode(userMessage);
    const typing = typingNode();
    ui.conversation.append(pendingUserNode, typing);
    ui.conversation.setAttribute("aria-busy", "true");
    ui.status.textContent = "Thinking about your question…";
    scrollToLatest();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 20000);
    try {
      const previousSession = state.sessionId;
      const response = await fetch("/api/chat", {
        method: "POST", headers: {"Content-Type": "application/json", Accept: "application/json"},
        body: JSON.stringify({message: text, session_id: state.sessionId, mode: state.mode}),
        signal: controller.signal,
      });
      const result = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = typeof result.error === "string" ? result.error : result.error?.message;
        throw new Error(detail || "Your message could not be sent. Please try again.");
      }
      if (typeof result.reply !== "string" || typeof result.session_id !== "string") throw new Error("Olive’s reply was incomplete. Please try again.");
      typing.remove();
      if (previousSession && previousSession !== result.session_id) {
        const notice = {role: "system", text: "Your previous session expired. This reply starts a new conversation."};
        state.messages.push(notice);
        pendingUserNode.before(messageNode(notice));
      }
      state.sessionId = result.session_id;
      const assistantMessage = {role: "assistant", text: result.reply, metadata: {
        method: result.method, intent: result.intent, confidence: result.confidence, sources: result.sources,
        context: result.context, elapsed_ms: result.elapsed_ms,
      }};
      state.messages.push(userMessage, assistantMessage);
      state.messages = state.messages.slice(-MAX_MESSAGES);
      ui.conversation.append(messageNode(assistantMessage));
      state.prompts = Array.isArray(result.suggested_prompts) ? result.suggested_prompts.filter((p) => typeof p === "string").slice(0, 3) : INITIAL_PROMPTS;
      ui.input.value = "";
      renderPrompts();
      saveSession();
      scrollToLatest();
    } catch (error) {
      typing.remove();
      pendingUserNode.remove();
      // Retain the message in the composer so retrying does not lose any work.
      ui.input.value = text;
      const message = error.name === "AbortError" ? "That took a little too long. Your message is still here; please try again." : (error.message || "We couldn’t reach Olive. Your message is still here; please try again.");
      setError(message);
    } finally {
      clearTimeout(timeout);
      ui.conversation.removeAttribute("aria-busy");
      state.busy = false;
      ui.status.textContent = "Your restaurant concierge";
      sizeInput();
      ui.input.focus({preventScroll: true});
    }
  }

  async function resetConversation() {
    if (state.busy) return;
    const oldSession = state.sessionId;
    state.sessionId = null;
    state.messages = [];
    state.prompts = Array.isArray(state.config?.quick_prompts) ? state.config.quick_prompts.filter((p) => typeof p === "string").slice(0, 3) : INITIAL_PROMPTS;
    ui.input.value = "";
    setError("");
    renderConversation();
    renderPrompts();
    sizeInput();
    saveSession();
    announce("A new conversation is ready.");
    if (oldSession) {
      // Reset the old server context as well; an already-expired session is harmless.
      fetch(`/api/sessions/${encodeURIComponent(oldSession)}`, {method: "DELETE"}).catch(() => {});
    }
    if (state.ready) ui.input.focus({preventScroll: true});
  }

  function exportConversation() {
    if (!state.messages.length) return;
    const lines = ["OLIVE TABLE", "Restaurant concierge conversation", "Fictional restaurant · Coursework prototype", `Answer method: ${state.mode}`, ""];
    state.messages.forEach((message) => {
      lines.push(message.role === "user" ? "YOU" : message.role === "assistant" ? "OLIVE" : "NOTICE", message.text, "");
    });
    const blob = new Blob([lines.join("\n")], {type: "text/plain;charset=utf-8"});
    const url = URL.createObjectURL(blob);
    const download = element("a");
    download.href = url;
    download.download = `olive-table-conversation-${new Date().toISOString().slice(0, 10)}.txt`;
    document.body.append(download);
    download.click();
    download.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    announce("Your conversation has been downloaded.");
  }

  ui.form.addEventListener("submit", sendMessage);
  ui.input.addEventListener("input", sizeInput);
  ui.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      ui.form.requestSubmit();
    }
  });
  ui.reset.addEventListener("click", resetConversation);
  ui.export.addEventListener("click", exportConversation);
  ui.retry.addEventListener("click", loadConfig);
  ui.mode.addEventListener("change", () => {
    state.mode = ui.mode.value === "baseline" ? "baseline" : "enhanced";
    resetConversation();
  });
  $("menu-search").addEventListener("input", (event) => {
    state.search = event.target.value.trim().toLowerCase();
    renderMenu();
  });
  $("menu-category").addEventListener("change", (event) => {
    state.category = event.target.value;
    renderMenu();
  });
  $("dish-dialog-ask").addEventListener("click", () => {
    const dish = state.selectedDish;
    if (!dish || state.busy || !state.ready) return;
    $("dish-dialog").close();
    sendPrompt(`Tell me about the ${dish.name}.`);
  });
  $("dish-dialog").addEventListener("click", (event) => {
    if (event.target === $("dish-dialog")) $("dish-dialog").close();
  });
  document.querySelectorAll(".filter-button").forEach((button) => {
    button.addEventListener("click", () => {
      state.filter = button.dataset.filter;
      document.querySelectorAll(".filter-button").forEach((candidate) => {
        const selected = candidate === button;
        candidate.classList.toggle("active", selected);
        candidate.setAttribute("aria-pressed", String(selected));
      });
      renderMenu();
    });
  });
  restoreSession();
  renderConversation();
  renderPrompts();
  sizeInput();
  loadConfig();
})();
