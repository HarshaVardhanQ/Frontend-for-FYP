document.addEventListener("DOMContentLoaded", function () {
    // Panels and Resizer
    const formPanel = document.getElementById("formPanel");
    const resultPanel = document.getElementById("resultPanel");
    const resizer = document.getElementById("resizer");
  
    // Elements for prediction results inside resultPanel
    const predictionText = document.getElementById("predictionText");
    const stageExplanationText = document.getElementById("stageExplanationText");
    const featureExplanationsText = document.getElementById("featureExplanationsText");
    const calculatedValuesText = document.getElementById("calculatedValuesText");
    const additionalInfoText = document.getElementById("additionalInfoText");
    const closeButton = document.getElementById("closeButton");
    const predictionForm = document.getElementById("predictionForm");
  
    // Numeric limits (for optional validation)
    const limits = {
      age: [18, 90],
      bilirubin: [0.1, 15.0],
      alk_phos: [40, 550],
      alt: [7, 190],
      ast: [8, 250],
      albumin: [1.5, 5.0],
      proteins: [2.0, 7.9],
      prothrombin: [9.4, 35.0],
      platelets: [55, 450]
    };
  
    // Resizer logic: allow dragging to adjust formPanel width
    let isResizing = false;
    let startX = 0;
    let startWidth = 0;
  
    resizer.addEventListener("mousedown", function(e) {
      e.preventDefault();
      isResizing = true;
      startX = e.clientX;
      startWidth = formPanel.getBoundingClientRect().width;
      document.addEventListener("mousemove", onMouseMove);
      document.addEventListener("mouseup", onMouseUp);
    });
  
    function onMouseMove(e) {
      if (!isResizing) return;
      const dx = e.clientX - startX;
      let newWidth = startWidth + dx;
      // Clamp the width of formPanel
      if (newWidth < 250) newWidth = 250;
      if (newWidth > 700) newWidth = 700;
      formPanel.style.width = newWidth + "px";
    }
  
    function onMouseUp() {
      isResizing = false;
      document.removeEventListener("mousemove", onMouseMove);
      document.removeEventListener("mouseup", onMouseUp);
    }
  
    // Optional: Validation functions
    function validateInput(input) {
      const fieldName = input.name;
      const rawValue = input.value.trim();
      if (limits[fieldName]) {
        if (!/^\d+(\.\d+)?$/.test(rawValue)) {
          input.style.border = "2px solid red";
          showError(input, "Please enter a valid numeric value.");
          return false;
        }
        const numericValue = parseFloat(rawValue);
        const [min, max] = limits[fieldName];
        if (numericValue < min || numericValue > max) {
          input.style.border = "2px solid red";
          showError(input, `Value must be between ${min} and ${max}`);
          return false;
        } else {
          input.style.border = "1px solid #ccc";
          removeError(input);
          return true;
        }
      }
      removeError(input);
      input.style.border = "1px solid #ccc";
      return true;
    }
  
    function showError(input, message) {
      let errorMsg = input.parentNode.querySelector(".error-message");
      if (!errorMsg) {
        errorMsg = document.createElement("p");
        errorMsg.className = "error-message";
        errorMsg.style.color = "red";
        errorMsg.style.fontSize = "0.85rem";
        errorMsg.style.marginTop = "5px";
        input.parentNode.appendChild(errorMsg);
      }
      errorMsg.textContent = message;
    }
  
    function removeError(input) {
      const errorMsg = input.parentNode.querySelector(".error-message");
      if (errorMsg) {
        errorMsg.remove();
      }
    }
  
    document.querySelectorAll("input").forEach(input => {
      input.addEventListener("input", () => validateInput(input));
    });
  
    // Form submission logic
    predictionForm.addEventListener("submit", function (event) {
      event.preventDefault();
  
      let isValid = true;
      document.querySelectorAll("input").forEach(input => {
        if (!validateInput(input)) isValid = false;
      });
      if (!isValid) {
        alert("⚠️ Please enter valid values before submitting!");
        return;
      }
  
      // Show loading
      document.getElementById("loadingIndicator").parentNode.classList.remove("hidden");
  
      // Gather form data
      const formData = new FormData(predictionForm);
      const jsonData = {};
      const fieldMapping = {
        age: "Age",
        Gender: "Gender",
        bilirubin: "Total Bilirubin",
        alk_phos: "Alkaline Phosphatase",
        alt: "Alanine Aminotransferase",
        ast: "Aspartate Aminotransferase",
        albumin: "Albumin",
        proteins: "Total Proteins",
        prothrombin: "Prothrombin Time",
        platelets: "Platelets",
        Ascites: "Ascites",
        LiverFirmness: "LiverFirmness"
      };
  
      formData.forEach((value, key) => {
        const correctedKey = fieldMapping[key] || key;
        if (correctedKey === "Gender") {
          jsonData[correctedKey] = value.toLowerCase() === "male" ? "Male" : "Female";
        } else if (correctedKey === "Ascites" || correctedKey === "LiverFirmness") {
          jsonData[correctedKey] = value.toLowerCase() === "present" ? "Present" : "Absent";
        } else {
          const numValue = parseFloat(value);
          jsonData[correctedKey] = isNaN(numValue) ? 0 : numValue;
        }
      });
      console.log("✅ Sending Data:", jsonData);
  
      // Fetch prediction from backend
      fetch("https://backend-only-for-fyp-production.up.railway.app/", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(jsonData)
      })
      .then(response => {
        console.log("🔄 API Response Status:", response.status);
        if (!response.ok) {
          throw new Error(`HTTP Error! Status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        document.getElementById("loadingIndicator").parentNode.classList.add("hidden");
        console.log("✅ API Response Data:", data);
  
        if (data.error) {
          predictionText.innerText = "❌ Error: " + data.error;
          stageExplanationText.innerHTML = "";
          featureExplanationsText.innerHTML = "";
          calculatedValuesText.innerHTML = "";
          additionalInfoText.innerHTML = "";
          resultPanel.classList.remove("hidden");
          resultPanel.classList.remove("show");
          return;
        }
  
        predictionText.innerText = `Prediction: ${data["Predicted Stage"] || "Unknown"}`;
        stageExplanationText.innerHTML = data["Stage Explanation"]
          ? `<strong>Stage Explanation:</strong><br>${data["Stage Explanation"]}`
          : "";
  
        if (data["Feature Explanations"] && Array.isArray(data["Feature Explanations"])) {
          const bulletItems = data["Feature Explanations"].map(item => `<li>${item}</li>`).join("");
          featureExplanationsText.innerHTML = `<strong>Feature Explanations:</strong><br><ul>${bulletItems}</ul>`;
        } else {
          featureExplanationsText.innerHTML = "";
        }
  
        if (data["Calculated Values"]) {
          const calcVals = data["Calculated Values"];
          calculatedValuesText.innerHTML = `<strong>Calculated Values:</strong><br>
            AST/ALT Ratio: ${calcVals["AST ALT Ratio"]}<br>
            FIB-4 Score: ${calcVals["FIB-4 Score"]}<br>
            Albumin Globulin Ratio: ${calcVals["Albumin Globulin Ratio"]}`;
        } else {
          calculatedValuesText.innerHTML = "";
        }
  
        additionalInfoText.innerHTML = `<strong>Additional Info:</strong><br>
          Ascites: ${data["Ascites"]}<br>
          Liver Firmness: ${data["LiverFirmness"]}`;
  
        // Show the result panel
        resultPanel.classList.remove("hidden");
        resultPanel.classList.add("show");
      })
      .catch(error => {
        document.getElementById("loadingIndicator").parentNode.classList.add("hidden");
        console.error("❌ Fetch Error:", error);
        predictionText.innerText = "❌ Error: Unable to fetch prediction";
        stageExplanationText.innerHTML = "";
        featureExplanationsText.innerHTML = "";
        calculatedValuesText.innerHTML = "";
        additionalInfoText.innerHTML = "";
        resultPanel.classList.remove("hidden");
        resultPanel.classList.remove("show");
      });
    });
  
    // Close button hides result panel
    closeButton.addEventListener("click", function () {
      resultPanel.classList.remove("show");
      resultPanel.classList.add("hidden");
    });
  });
  
