(function () {
  const video = document.getElementById("liveVideo");
  const canvas = document.getElementById("capCanvas");
  const btnStart = document.getElementById("btnStart");
  const btnStop = document.getElementById("btnStop");
  const btnSave = document.getElementById("btnSaveFrame");
  const statusLine = document.getElementById("statusLine");
  const predSign = document.getElementById("predSign");
  const predConf = document.getElementById("predConf");
  const textOut = document.getElementById("textOut");
  const csrf = document.querySelector('meta[name="csrf-token"]')?.content || "";

  let stream = null;
  let intervalId = null;

  function setStatus(msg) {
    if (statusLine) statusLine.textContent = msg;
  }

  async function startCam() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      video.srcObject = stream;
      await video.play();
      btnStart.disabled = true;
      btnStop.disabled = false;
      btnSave.disabled = false;
      setStatus("Camera running. Show your hand in the frame.");
      intervalId = window.setInterval(sendFrame, 400);
    } catch (e) {
      setStatus("Camera error: " + (e.message || "Could not access webcam."));
    }
  }

  function stopCam() {
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      stream = null;
    }
    video.srcObject = null;
    btnStart.disabled = false;
    btnStop.disabled = true;
    btnSave.disabled = true;
    setStatus("Camera stopped.");
  }

  async function sendFrame() {
    if (!video.videoWidth) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);
    canvas.toBlob(async (blob) => {
      if (!blob) return;
      try {
        const res = await fetch("/api/predict-frame", {
          method: "POST",
          headers: {
            "Content-Type": "image/jpeg",
            "X-CSRFToken": csrf,
          },
          body: blob,
          credentials: "same-origin",
        });
        const data = await res.json();
        if (!data.ok) {
          setStatus(data.error || "Prediction failed.");
          return;
        }
        if (!data.hand_detected) {
          setStatus(data.status || "No hand detected — move hand inside frame.");
          predSign.textContent = "—";
          predConf.textContent = "—";
          return;
        }
        setStatus("Hand detected · Prediction in progress");
        predSign.textContent = data.sign ?? "—";
        predConf.textContent = data.confidence != null ? (data.confidence * 100).toFixed(1) + "%" : "—";
      } catch (e) {
        setStatus("Network error during prediction.");
      }
    }, "image/jpeg", 0.85);
  }

  btnStart?.addEventListener("click", startCam);
  btnStop?.addEventListener("click", stopCam);

  btnSave?.addEventListener("click", async () => {
    if (!video.videoWidth) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const res = await fetch("/save-capture", {
        method: "POST",
        headers: { "X-CSRFToken": csrf },
        body: blob,
        credentials: "same-origin",
      });
      if (res.redirected) window.location.href = res.url;
    }, "image/jpeg", 0.92);
  });

  document.getElementById("btnAdd")?.addEventListener("click", () => {
    const s = predSign.textContent.trim();
    if (s && s !== "—") textOut.value += s;
  });
  document.getElementById("btnSpace")?.addEventListener("click", () => {
    textOut.value += " ";
  });
  document.getElementById("btnDel")?.addEventListener("click", () => {
    textOut.value = textOut.value.slice(0, -1);
  });
  document.getElementById("btnClearText")?.addEventListener("click", () => {
    textOut.value = "";
  });
  document.getElementById("btnClearPred")?.addEventListener("click", () => {
    predSign.textContent = "—";
    predConf.textContent = "—";
  });
})();
