document.addEventListener("DOMContentLoaded", () => {
  const imageInput = document.getElementById("imageInput");
  const uploadedImage = document.getElementById("uploadedImage");
  const caption = document.getElementById("caption");
  const description = document.getElementById("description");

  let loadingInterval;

  function startLoadingAnimation(element, baseText = "Loading") {
    let dotCount = 0;
    loadingInterval = setInterval(() => {
      dotCount = (dotCount + 1) % 4;
      element.textContent = baseText + ".".repeat(dotCount);
    }, 500);
  }

  function stopLoadingAnimation(element, finalText) {
    clearInterval(loadingInterval);
    element.textContent = finalText;
  }

  imageInput.addEventListener("change", async () => {
    const file = imageInput.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      uploadedImage.src = e.target.result;
      uploadedImage.style.display = "block";
    };
    reader.readAsDataURL(file);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      const topPrediction = result.prediction?.[0]?.[0];

      const predictions = result.prediction;

      for (let i = 0; i < predictions.length; i++) {
        console.log(predictions[i])
        caption.textContent += predictions[i][0] + ": " + predictions[i][1] * 100 + "%" +  "\n";
        if(i != predictions.length-1){
          caption.textContent += "\n";
        }
      }

      // caption.textContent = `Prediction: ${JSON.stringify(result.prediction)}`;
      
      startLoadingAnimation(description, "Loading description");

      if (topPrediction) {
        const descResponse = await fetch(
          `http://localhost:8000/describe?plant_name=${encodeURIComponent(topPrediction)}`
        );
        const descResult = await descResponse.json();

        stopLoadingAnimation(description, descResult.message);
      } else {
        stopLoadingAnimation(description, "Description not found.");
      }
    } catch (error) {
      console.error("Upload or prediction failed:", error);
      caption.textContent = "Upload or prediction failed.";
      stopLoadingAnimation(description, "An error occurred while loading the description.");
    }
  });
});
