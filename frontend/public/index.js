document.addEventListener("DOMContentLoaded", () => {
  const imageInput = document.getElementById("imageInput");
  const uploadedImage = document.getElementById("uploadedImage");
  const caption = document.getElementById("caption");
  const description = document.getElementById("description")

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

      // Get top plant name from prediction list
      const topPrediction = result.prediction?.[0]?.[0];

      // Show prediction immediately
      caption.textContent = `Prediction: ${JSON.stringify(result.prediction)}`;
      description.textContent = "Loading description...";

      // If top prediction exists, call /describe
      if (topPrediction) {
        const descResponse = await fetch(`http://localhost:8000/describe?plant_name=${encodeURIComponent(topPrediction)}`);
        const descResult = await descResponse.json();

        // Append description
        caption.textContent = `Prediction: ${JSON.stringify(result.prediction)}'`;
        description.textContent = `${descResult.message}`;
      } else {
        description.textContent += `\n\nDescription not found.`;
      }
    } catch (error) {
      console.error("Upload or prediction failed:", error);
      caption.textContent = "Upload or prediction failed.";
    }
  });
});
