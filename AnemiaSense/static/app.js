const form = document.querySelector("#prediction-form");
const submitButton = form.querySelector("button[type='submit']");
const resultEmpty = document.querySelector("#result-empty");
const resultContent = document.querySelector("#result-content");
const formError = document.querySelector("#form-error");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  formError.textContent = "";
  submitButton.disabled = true;
  submitButton.querySelector("span:first-child").textContent = "Analyzing pattern...";

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(Object.fromEntries(new FormData(form))),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The estimate could not be generated.");

    document.querySelector("#result-label").textContent = result.label;
    document.querySelector("#result-probability").textContent = `${Math.round(result.probability * 100)}%`;
    document.querySelector("#confidence-fill").style.width = `${Math.round(result.probability * 100)}%`;
    document.querySelector("#result-notice").textContent = result.notice;
    resultEmpty.hidden = true;
    resultContent.hidden = false;
  } catch (error) {
    formError.textContent = error.message;
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector("span:first-child").textContent = "Run pattern analysis";
  }
});
