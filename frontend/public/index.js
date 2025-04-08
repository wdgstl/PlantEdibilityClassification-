document.addEventListener("DOMContentLoaded", () => {
  const imageInput = document.getElementById("imageInput");
  const uploadedImage = document.getElementById("uploadedImage");
  const caption = document.getElementById("caption");

  imageInput.addEventListener("change", async () => {
    const file = imageInput.files?.[0];
    if (!file) return;

    // Display image preview
    const reader = new FileReader();
    reader.onload = (e) => {
      uploadedImage.src = e.target.result;
      uploadedImage.style.display = "block";
    };
    reader.readAsDataURL(file);

    // Send a fetch request to your backend
    try {
      const res = await fetch("http://localhost:8000/add?x=10&y=20");
      const data = await res.json();
      caption.textContent = `Prediction: ${data.result}`;
    } catch (err) {
      console.error("Error during fetch:", err);
      caption.textContent = "Failed to get prediction.";
    }
  });
});
